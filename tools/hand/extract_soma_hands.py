"""Extract wrist-local `SOMAHandLayer` animations from full-body SOMA motions.

The SOMA body rig already contains each hand as a contiguous 25-joint subtree,
and ``SOMAHandLayer.hand_joint_ids_global`` names exactly which body joints
those are.  The within-hand parent structure is identical to the hand layer's
own ``joint_parent_ids``, so extracting a hand is a **re-index of the body pose,
not a re-fit** — the finger rotations are carried across unchanged.

The only quantity that needs evaluating is the wrist's world transform, which
depends on the whole arm chain; it comes from ``SOMALayer.pose()``'s
``transforms`` output.  It is stored alongside the hand motion so a consumer can
put the hand back into world space, while ``poses``/``transl`` themselves
describe the hand in its own wrist frame.

Works on any full-body SOMA motion written by ``soma.io.save_soma_npz`` that
carries a personalized subject template, so it applies to converted datasets and
to generated motion alike.

Usage::

    python -m tools.hand.extract_soma_hands \\
        --package ../GRAB_SOMA_51Objects --output-dir ../GRAB_SOMA_Hands

    # one sequence, for inspection
    python -m tools.hand.extract_soma_hands --package ... --output-dir ... \\
        --motion motions/s1/mug_drink_1.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from soma.body import SOMALayer
from soma.hand import SOMAHandLayer
from soma.io import save_soma_npz

HANDS = ("left", "right")


_BODY_LAYER_CACHE: dict[tuple[str, str], SOMALayer] = {}


def build_body_layer(package: Path, motion: dict, assets: Path, device: str) -> SOMALayer:
    """The subject's personalized body layer (mirrors tools/replay_grab_soma.py).

    Cached per (template, gender): a dataset has a handful of subjects but
    thousands of sequences, and rebuilding the layer dominates runtime.
    """
    template_path = str(motion["subject_template_path"].item())
    gender = str(motion["gender"].item())
    key = (template_path, gender)
    if key in _BODY_LAYER_CACHE:
        return _BODY_LAYER_CACHE[key]
    template = trimesh.load(package / template_path,
                            process=False, maintain_order=True)
    layer = SOMALayer(assets, lod="low", identity_model_type="smplx",
                      identity_model_kwargs={"gender": gender},
                      device=device, mode="dense" if device == "cpu" else "warp")
    layer.identity_model.identity_model.v_template = torch.as_tensor(
        np.asarray(template.vertices), dtype=torch.float32, device=device)
    layer.prepare_identity(torch.zeros(1, 10, device=device))
    _BODY_LAYER_CACHE[key] = layer
    return layer


def wrist_transforms(layer: SOMALayer, motion: dict, wrist_ids: dict[str, int],
                     device: str, batch: int) -> dict[str, np.ndarray]:
    """World 4x4 transform of each wrist, per frame.

    ``SOMALayer.pose`` takes poses without the virtual Root but returns
    ``transforms`` *with* it, so transform indices match the rig's joint order.
    """
    poses = motion["poses"]
    transl = motion["transl"]
    out = {side: np.empty((len(poses), 4, 4), dtype=np.float64) for side in wrist_ids}
    for start in range(0, len(poses), batch):
        stop = min(start + batch, len(poses))
        with torch.no_grad():
            # Slicing off the virtual Root leaves a non-contiguous array, which
            # the layer's internal view() calls reject.
            chunk = np.ascontiguousarray(poses[start:stop, 1:], dtype=np.float32)
            res = layer.pose(
                torch.as_tensor(chunk, device=device),
                transl=torch.as_tensor(np.ascontiguousarray(transl[start:stop], dtype=np.float32),
                                       device=device),
                pose2rot=True, absolute_pose=True, apply_correctives=False)
        tf = res["transforms"].detach().cpu().numpy()
        for side, jid in wrist_ids.items():
            out[side][start:stop] = tf[:, jid]
    return out


def extract_hand(motion: dict, joint_ids: np.ndarray, wrist_world: np.ndarray,
                 world_frame: bool):
    """Return (poses (T,25,3), transl (T,3)) for one hand."""
    hand_poses = np.asarray(motion["poses"], dtype=np.float64)[:, joint_ids].copy()

    if world_frame:
        # Root carries the wrist's world orientation; translation places it.
        hand_poses[:, 0] = Rotation.from_matrix(wrist_world[:, :3, :3]).as_rotvec()
        transl = wrist_world[:, :3, 3].copy()
    else:
        # Wrist-local: the hand sits at the origin, unrotated. Finger rotations
        # are parent-relative and so are unchanged by the change of frame.
        hand_poses[:, 0] = 0.0
        transl = np.zeros((len(hand_poses), 3), dtype=np.float64)
    return hand_poses, transl


def object_in_wrist_frame(motion: dict, wrist_world: np.ndarray):
    """Carry the object track into the wrist frame (a rigid change of basis)."""
    if "object_trans" not in motion or "object_rot" not in motion:
        return None
    o_t = np.asarray(motion["object_trans"], dtype=np.float64)
    o_r = Rotation.from_rotvec(np.asarray(motion["object_rot"], dtype=np.float64)).as_matrix()
    w_R = wrist_world[:, :3, :3]
    w_t = wrist_world[:, :3, 3]
    inv = np.transpose(w_R, (0, 2, 1))
    return (np.einsum("tab,tb->ta", inv, o_t - w_t),
            Rotation.from_matrix(inv @ o_r).as_rotvec())


def carry_over(motion: dict, keys) -> dict:
    return {k: motion[k] for k in keys if k in motion}


def process_motion(path: Path, package: Path, out_dir: Path, layers, assets: Path,
                   device: str, batch: int, world_frame: bool) -> list[Path]:
    motion = dict(np.load(path, allow_pickle=False))
    body = build_body_layer(package, motion, assets, device)

    wrist_ids = {s: int(layers[s].hand_joint_ids_global[0]) for s in layers}
    wrists = wrist_transforms(body, motion, wrist_ids, device, batch)

    rel = path.relative_to(package / "motions")
    written = []
    for side, layer in layers.items():
        ids = np.asarray([int(i) for i in layer.hand_joint_ids_global])
        # Keep the body rig's own names so each hand joint stays traceable.
        hand_joint_names = [str(n) for n in np.asarray(motion["joint_names"])[ids]]
        poses, transl = extract_hand(motion, ids, wrists[side], world_frame)

        extra = carry_over(motion, (
            "fps", "source_dataset", "source_sequence", "subject_id", "gender",
            "motion_intent", "object_id", "package_mesh_path", "table_trans",
            "table_rot", "package_table_mesh_path", "fit_mean_vertex_error_m"))
        extra["wrist_world_rotation"] = Rotation.from_matrix(
            wrists[side][:, :3, :3]).as_rotvec().astype(np.float32)
        extra["wrist_world_translation"] = wrists[side][:, :3, 3].astype(np.float32)
        extra["source_body_motion"] = str(rel)
        extra["hand_frame"] = "world" if world_frame else "wrist_local"
        extra["body_joint_ids"] = ids.astype(np.int32)

        obj = object_in_wrist_frame(motion, wrists[side])
        if obj is not None:
            extra["object_trans"] = obj[0].astype(np.float32)
            extra["object_rot"] = obj[1].astype(np.float32)
            extra["object_frame"] = "wrist_local"

        dest = out_dir / side / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_soma_npz(
            dest,
            poses=poses.astype(np.float32),
            transl=transl.astype(np.float32),
            joint_names=hand_joint_names,
            identity_model_type="soma",
            identity_coeffs=np.zeros((1, layer.num_shape_components), dtype=np.float32),
            hand_type=side,
            unit="meters",
            keep_root=True,
            extra_arrays=extra,
        )
        written.append(dest)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    ap.add_argument("--motion", default=None,
                    help="single motion path relative to the package")
    ap.add_argument("--hands", default="left,right")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-sequences", type=int, default=0)
    ap.add_argument("--world-frame", action="store_true",
                    help="place the hand in world space instead of its wrist frame")
    ap.add_argument("--resume", action="store_true", help="skip existing outputs")
    args = ap.parse_args()

    sides = [s.strip() for s in args.hands.split(",") if s.strip()]
    for s in sides:
        if s not in HANDS:
            raise SystemExit(f"--hands must be from {HANDS}, got {s!r}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        args.device = "cpu"

    layers = {s: SOMAHandLayer(data_root=args.assets, hand_type=s,
                               identity_model_type="soma", device="cpu") for s in sides}

    if args.motion:
        motions = [args.package / args.motion]
    else:
        motions = sorted((args.package / "motions").rglob("*.npz"))
    if args.max_sequences:
        motions = motions[:args.max_sequences]

    done = 0
    for i, path in enumerate(motions, 1):
        rel = path.relative_to(args.package / "motions")
        if args.resume and all((args.output_dir / s / rel).exists() for s in sides):
            continue
        process_motion(path, args.package, args.output_dir, layers,
                       args.assets, args.device, args.batch_size, args.world_frame)
        done += 1
        if done % 25 == 0 or i == len(motions):
            print(f"  {i}/{len(motions)} sequences", flush=True)

    manifest = {
        "source_package": str(args.package),
        "hands": sides,
        "frame": "world" if args.world_frame else "wrist_local",
        "joints_per_hand": 25,
        "sequences": done,
        "note": ("Finger rotations are copied from the body rig unchanged: the SOMA body "
                 "skeleton contains each hand as a 25-joint subtree whose parent structure "
                 "matches SOMAHandLayer exactly, so no re-fitting is involved. Hand *shape* "
                 "is the generic SOMA hand identity with zero coefficients - the body's "
                 "personalized SMPL-X template is not transferable to the hand layer, which "
                 "supports soma/mano/mhr identities only."),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {done} sequences x {len(sides)} hand(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
