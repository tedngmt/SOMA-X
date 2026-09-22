"""Convert GRIP hand-motion results into clips for the Unity SOMA avatar.

GRIP (github.com/otaheri/GRIP) writes one ``.pt`` per GRAB sequence when run with
``--out-type params``: for every frame the recorded SMPL-X pose, the pose with
GRIP's coarse hands (CNet) and the pose with its refined hands (RNet), plus the
object track and the subject's personalized ``v_template``.

Each requested variant goes through the same route GRAB itself takes in
``convert_grab_to_soma.py`` -- SMPL-X vertices, then ``PoseInversion`` onto the
SOMA rig -- so recorded and generated hands are compared on equal terms.

The Unity avatar (soma-neutral.fbx) binds every bone at identity in a T-pose, so
a bone's rotation there is its world-space change from the T-pose, expressed
parent-relative.  That is ``W_j(t) @ T_j^T`` from SOMA's forward kinematics.
GRAB is right-handed Z-up and Unity is left-handed Y-up; one reflection
``(x, y, z) -> (x, z, y)`` carries positions across and conjugates rotations.

Output, read by Assets/Hand/Grip/GripClipPlayer.cs::

    <out>/<sequence>__<variant>.json   Hips world pose + 76 parent-local rotations
    <out>/<object>.mesh.json           object mesh, already in Unity's frame

This is the middle step of the pipeline folder, ``$MOGENVR_ROOT`` (default
``~/Projects/MoGenVR_Pipeline``), and every path defaults to its place in that layout::

    <root>/GRIP/grab/          object meshes           --grab
    <root>/SOMA-X/input/         GRIP's .pt results      --grip-results  (searched recursively)
    <root>/MoGenVR_Unity/GripClips/   the clips        --out

Each written clip is also copied into the live Unity project's ``GripClips/`` when that
folder exists (``--unity``; ``--no-unity`` to skip), so Unity sees new clips without a
manual copy. Run it through ``<root>/SOMA-X/run_convert.sh``, which activates the ``soma`` env.

Usage::

    python -m tools.convert_grip_to_unity                      # everything in SOMA-X/input
    python -m tools.convert_grip_to_unity --seq s1_mug_drink_1  # one sequence
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soma.body import SOMALayer  # noqa: E402
from soma.fitting.pose_inversion import PoseInversion  # noqa: E402

FPS = 30  # GRIP's process_data.py keeps every 4th frame of GRAB's 120 fps

# GRAB (right-handed, Z up) -> Unity (left-handed, Y up): swap Y and Z.
M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
# SOMA's own T-pose frame (right-handed, Y up) -> Unity: mirror Z. The T-pose lives in
# this frame while the posed motion lives in GRAB's, so a bone's change from the T-pose
# takes M on the posed side and A on the T-pose side.
A = np.diag([1.0, 1.0, -1.0])

# Which entry of the GRIP result holds the pose for each variant.
VARIANTS = {"gt": "params_gt", "coarse": "bparams", "grip": "bparams_ref"}


def first_frames(chunks, key, tail):
    """GRIP saves one entry per frame holding that frame and the next one, so entries
    overlap by a frame. Take the first of each to get the sequence back."""
    return torch.stack([c[key].detach().cpu().reshape((2,) + tuple(tail))[0] for c in chunks])


def load_variant(data, variant):
    chunks = data[VARIANTS[variant]]
    pose = first_frames(chunks, "fullpose_rotmat", (55, 3, 3))
    transl = first_frames(chunks, "transl", (3,))
    return pose.float(), transl.float()


def smplx_vertices(model, pose, transl, device, batch):
    verts = []
    for s in range(0, pose.shape[0], batch):
        p = pose[s:s + batch].to(device)
        with torch.no_grad():
            out = model(global_orient=p[:, 0:1], body_pose=p[:, 1:22], jaw_pose=p[:, 22:23],
                        leye_pose=p[:, 23:24], reye_pose=p[:, 24:25],
                        left_hand_pose=p[:, 25:40], right_hand_pose=p[:, 40:55],
                        transl=transl[s:s + batch].to(device))
        verts.append(out.vertices)
    return torch.cat(verts, dim=0)


def to_unity(world, t_orient, parents, hips):
    """SOMA world transforms (T, J, 4, 4) -> Hips position, per-bone quaternions.

    Bone 0 of the result is Hips with a WORLD rotation; the rest are parent-local.
    The Root joint (index 0 in SOMA) has no counterpart on the Unity rig.
    """
    delta = world[:, :, :3, :3] @ np.transpose(t_orient, (0, 2, 1))[None]   # from the T-pose
    delta = M[None, None] @ delta @ A[None, None]
    T, J = delta.shape[:2]
    quats = np.empty((T, J - 1, 4))
    for j in range(1, J):
        if j == hips:
            r = delta[:, j]
        else:
            r = np.transpose(delta[:, parents[j]], (0, 2, 1)) @ delta[:, j]
        quats[:, j - 1] = Rotation.from_matrix(r).as_quat()   # (x, y, z, w)
    hips_pos = world[:, hips, :3, 3] @ M.T
    return hips_pos, quats


def active_range(world, names, rest_deg):
    """Frames between the T-poses that open and close every GRAB recording.

    Upper-arm elevation is the angle between shoulder->elbow and straight down (GRAB is
    Z up): about 90 degrees in the T-pose, 10-20 with the arms hanging. Keep from the first
    frame both arms are down to the last one; anything in between, a raised drinking arm
    included, stays.
    """
    down = np.array([0.0, 0.0, -1.0])
    lowered = np.ones(world.shape[0], dtype=bool)
    for side in ("Left", "Right"):
        upper = world[:, names.index(side + "ForeArm"), :3, 3] - world[:, names.index(side + "Arm"), :3, 3]
        upper /= np.linalg.norm(upper, axis=1, keepdims=True) + 1e-9
        lowered &= np.degrees(np.arccos(np.clip(upper @ down, -1.0, 1.0))) < rest_deg
    idx = np.flatnonzero(lowered)
    if len(idx) < 2:
        return 0, world.shape[0]
    return int(idx[0]), int(idx[-1]) + 1


def fk_check(world, t_world, t_orient, parents, hips, hips_pos, quats):
    """Rebuild joint positions the way Unity will and compare with SOMA's own."""
    T, J = world.shape[:2]
    rest = np.zeros((J, 3))
    for j in range(1, J):
        if j != hips:
            rest[j] = (t_world[j, :3, 3] - t_world[parents[j], :3, 3]) @ A
    g_rot = np.empty((T, J, 3, 3))
    g_pos = np.empty((T, J, 3))
    order = [hips] + [j for j in range(1, J) if j != hips]
    for j in order:
        local = Rotation.from_quat(quats[:, j - 1]).as_matrix()
        if j == hips:
            g_rot[:, j], g_pos[:, j] = local, hips_pos
        else:
            p = parents[j]
            g_rot[:, j] = g_rot[:, p] @ local
            g_pos[:, j] = g_pos[:, p] + np.einsum("tab,b->ta", g_rot[:, p], rest[j])
    want = world[:, 1:, :3, 3] @ M.T
    return np.linalg.norm(g_pos[:, 1:] - want, axis=-1)


def object_track(data, grab_root):
    chunks = data["obj_params"]
    transl = first_frames(chunks, "transl", (3,)).numpy()
    orient = first_frames(chunks, "global_orient", (3,)).numpy()      # axis-angle
    rot = Rotation.from_rotvec(orient).as_matrix()
    # GRAB poses objects as ``verts @ R + t`` (row vectors), so the column-vector
    # rotation is R^T.
    rot = np.transpose(rot, (0, 2, 1))
    rot = M[None] @ rot @ M.T[None]
    return transl @ M.T, Rotation.from_matrix(rot).as_quat()


def write_mesh(grab_root, name, out_dir):
    target = out_dir / f"{name}.mesh.json"
    if target.exists():
        return target.name
    mesh = trimesh.load(grab_root / "tools/object_meshes/contact_meshes" / f"{name}.ply",
                        process=False)
    verts = np.asarray(mesh.vertices) @ M.T
    faces = np.asarray(mesh.faces)[:, ::-1]      # the reflection flips the winding
    target.write_text(json.dumps({"vertices": np.round(verts, 5).reshape(-1).tolist(),
                                  "triangles": faces.reshape(-1).tolist()},
                                 separators=(",", ":")), encoding="utf-8")
    return target.name


PIPELINE = Path(os.environ.get("MOGENVR_ROOT", "~/Projects/MoGenVR_Pipeline")).expanduser()
UNITY_CLIPS = Path("/mnt/c/Unity Project/MoGenVR_Unity/GripClips")


def unity_dir(args):
    """The live Unity project's clip folder, or None when it is absent or switched off."""
    if args.no_unity or args.unity is None:
        return None
    if not args.unity.parent.is_dir():      # the project itself is not on this machine
        return None
    args.unity.mkdir(parents=True, exist_ok=True)
    return args.unity


def mirror(path, dest):
    """Copy one written file into dest, unless dest already has it at least as new."""
    if dest is None or path is None or not path.exists():
        return
    dst = dest / path.name
    if dst.exists() and dst.stat().st_mtime >= path.stat().st_mtime:
        return
    shutil.copy2(path, dst)


def find_results(folder):
    """GRIP's .pt files under folder, searched recursively; a run's snapshots/ is skipped."""
    files = sorted(folder.glob("*.pt"))
    if not files:
        files = sorted(p for p in folder.rglob("*.pt") if "snapshots" not in p.parts)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grip-results", type=Path, default=PIPELINE / "SOMA-X" / "input",
                    help="folder of GRIP <sequence>.pt files, searched recursively "
                         "(default: <root>/SOMA-X/input)")
    ap.add_argument("--out", type=Path, default=PIPELINE / "MoGenVR_Unity" / "GripClips",
                    help="where the clips go (default: <root>/MoGenVR_Unity/GripClips)")
    ap.add_argument("--grab", type=Path, default=PIPELINE / "GRIP" / "grab",
                    help="GRAB root holding tools/object_meshes/contact_meshes "
                         "(default: <root>/GRIP/grab)")
    ap.add_argument("--assets", type=Path, default=ROOT / "assets")
    ap.add_argument("--seq", action="append", default=None,
                    help="sequence name without .pt; repeatable (default: all)")
    ap.add_argument("--variants", default="gt,grip",
                    help="comma list from: " + ", ".join(VARIANTS))
    ap.add_argument("--keep-tpose", action="store_true",
                    help="keep the T-poses that open and close every GRAB recording")
    ap.add_argument("--rest-deg", type=float, default=35.0,
                    help="upper-arm elevation below which the arms count as lowered")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--unity", type=Path, default=UNITY_CLIPS,
                    help="live Unity project's GripClips/ to also copy each clip into "
                         f"(default: {UNITY_CLIPS}; skipped if the project is not there)")
    ap.add_argument("--no-unity", action="store_true", help="do not copy into the Unity project")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    share = unity_dir(args)
    print(f"pipeline root {PIPELINE}\n  results  {args.grip_results}\n  objects  {args.grab}"
          f"\n  clips -> {args.out}" + (f"\n  copy  -> {share}" if share else ""), flush=True)
    files = find_results(args.grip_results)
    if args.seq:
        files = [f for f in files if f.stem in set(args.seq)]
    if not files:
        raise SystemExit(f"no GRIP results matched in {args.grip_results}")

    rigs = {}
    for f in files:
        data = torch.load(f, map_location="cpu")
        gender = "female" if int(torch.as_tensor(data["gender"]).reshape(-1)[0]) == 2 else "male"
        subject = str(data["sbj_id"])
        vtemp = torch.as_tensor(data["sbj_vtemp"]).detach().cpu().reshape(-1, 10475, 3)[0].float()

        if subject not in rigs:
            model = smplx.SMPLXLayer(str(args.assets / "SMPLX" / f"SMPLX_{gender.upper()}.npz"),
                                     gender=gender, v_template=vtemp.to(args.device),
                                     use_pca=False, flat_hand_mean=True).to(args.device)
            soma = SOMALayer(args.assets, lod="low", identity_model_type="smplx",
                             identity_model_kwargs={"gender": gender},
                             device=args.device, mode="warp")
            soma.identity_model.identity_model.v_template = vtemp.to(args.device)
            inv = PoseInversion(soma, low_lod=False)
            with torch.no_grad():
                inv.prepare_identity(torch.zeros(1, 10, device=args.device))
            view = soma.public_rig_view()
            rigs[subject] = (model, soma, inv, view)
        model, soma, inv, view = rigs[subject]

        names = list(view.joint_names)
        parents = view.joint_parent_ids.detach().cpu().numpy().tolist()
        hips = int(soma.root_joint_idx)
        # This subject's own T-pose, in the same units as every posed frame. The rig
        # view's t_pose_world is the generic template (and in centimetres).
        with torch.no_grad():
            t_world = soma.pose(torch.zeros(1, len(names) - 1, 3, device=args.device),
                                pose2rot=True, absolute_pose=False,
                                fk_only=True)["transforms"][0].detach().cpu().numpy()
        t_orient = t_world[:, :3, :3]

        obj_name = str(data["obj_id"])
        obj_pos, obj_quat = object_track(data, args.grab)
        mesh_file = write_mesh(args.grab, obj_name, args.out)
        mirror(args.out / mesh_file, share)

        for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
            pose, transl = load_variant(data, variant)
            verts = smplx_vertices(model, pose, transl, args.device, args.batch_size)
            rots, roots, errs = [], [], []
            for s in range(0, verts.shape[0], args.batch_size):
                fit = inv.fit(verts[s:s + args.batch_size], body_iters=2, finger_iters=1,
                              full_iters=1, lie_iters=3, batch_size=args.batch_size)
                rots.append(fit["rotations"].detach())
                roots.append(fit["root_translation"].detach())
                errs.append(fit["per_vertex_error"].detach().mean(dim=-1))
            rots, roots = torch.cat(rots), torch.cat(roots)
            with torch.no_grad():
                world = soma.pose(rots[:, 1:], pose2rot=False, absolute_pose=True, transl=roots,
                                  fk_only=True)["transforms"].detach().cpu().numpy()

            hips_pos, quats = to_unity(world, t_orient, parents, hips)
            fk_err = fk_check(world, t_world, t_orient, parents, hips, hips_pos, quats)

            # Parent-relative bone offsets of THIS subject, so the avatar can take the
            # proportions the hand-object contacts were recorded with.
            rest = np.zeros((len(names) - 1, 3))
            for j in range(1, len(names)):
                if j != hips:
                    rest[j - 1] = (t_world[j, :3, 3] - t_world[parents[j], :3, 3]) @ A

            n = min(len(hips_pos), len(obj_pos))
            a, b = (0, n) if args.keep_tpose else active_range(world[:n], names, args.rest_deg)
            clip = {
                "name": f.stem, "source": variant, "fps": FPS, "frameCount": int(b - a),
                "sourceFrames": [int(a), int(b)],
                "bones": names[1:],
                "restPos": np.round(rest, 6).reshape(-1).tolist(),
                "hipsPos": np.round(hips_pos[a:b], 5).reshape(-1).tolist(),
                "rot": np.round(quats[a:b], 6).reshape(-1).tolist(),
                "obj": {"name": obj_name, "mesh": mesh_file,
                        "pos": np.round(obj_pos[a:b], 5).reshape(-1).tolist(),
                        "rot": np.round(obj_quat[a:b], 6).reshape(-1).tolist()},
            }
            n = b - a
            target = args.out / f"{f.stem}__{variant}.json"
            target.write_text(json.dumps(clip, separators=(",", ":")), encoding="utf-8")
            mirror(target, share)
            print(f"{target.name}: {n} frames | SOMA fit {float(torch.cat(errs).mean()) * 1000:.2f} mm"
                  f" | Unity FK check mean {fk_err.mean() * 1000:.3f} mm max {fk_err.max() * 1000:.3f} mm",
                  flush=True)


if __name__ == "__main__":
    main()
