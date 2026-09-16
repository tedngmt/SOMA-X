"""Inspect a paired source motion and optionally stage it for the official viewer.

Requires Python 3.9+ and NumPy. This helper does not install or launch a viewer.
Only load the trusted original GraspXL .npy files: their dictionaries use pickle.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


def package_file(root, relative):
    """Resolve a manifest path and require that it stays inside its package."""
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path leaves its package: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_destination(source, destination):
    if destination.exists() and (
        not destination.is_file() or sha256(source) != sha256(destination)
    ):
        raise FileExistsError(f"Refusing to replace different existing data: {destination}")


def copy_unchanged(source, destination):
    """Keep identical existing files; exclusive creation prevents overwriting."""
    check_destination(source, destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
    except FileExistsError:
        check_destination(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package", type=Path, default=Path(__file__).resolve().parent.parent,
        help="Raw MANO package root; defaults to the parent of scripts/",
    )
    parser.add_argument("--object-number", type=int, default=29, help="Manifest object number (mug: 29)")
    parser.add_argument("--sequence-index", type=int, default=0, help="Zero-based index within this object's motion_paths")
    parser.add_argument("--viewer-root", type=Path, help="Optional existing GraspXL_visualization repository")
    args = parser.parse_args()
    root = args.package.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    matches = [item for item in manifest["objects"] if item["number"] == args.object_number]
    if len(matches) != 1:
        parser.error(f"Object number {args.object_number} must identify one manifest entry")
    obj = matches[0]
    if not 0 <= args.sequence_index < len(obj["motion_paths"]):
        parser.error(f"--sequence-index must be between 0 and {len(obj['motion_paths']) - 1}")
    relative_mano = obj["motion_paths"][args.sequence_index]
    pairs = json.loads((root / "metadata" / "pairs.json").read_text(encoding="utf-8"))
    matching_pairs = [pair for pair in pairs if pair["mano_path"] == relative_mano]
    if len(matching_pairs) != 1:
        raise ValueError(f"Expected one SOMA pairing for {relative_mano}")
    pair = matching_pairs[0]
    if pair["object_key"] != obj["object_key"]:
        raise ValueError("Pair object_key differs from the selected manifest entry")
    mano_path = package_file(root, relative_mano)
    mesh_path = package_file(root, obj["mesh_path"])
    soma_root = (root / manifest["soma_package_relative"]).resolve()
    soma_path = package_file(soma_root, pair["soma_path"])
    raw = np.load(mano_path, allow_pickle=True).item()
    object_ids = [key for key in raw if key != "right_hand"]
    if object_ids != [obj["uid"]]:
        raise ValueError(f"Source object ID does not match the manifest: {object_ids}")
    object_id = object_ids[0]
    frames = pair["frames"]
    for name, widths in (
        ("right_hand", {"rot": 3, "trans": 3, "pose": 45}),
        (object_id, {"rot": 3, "trans": 3}),
    ):
        for field, width in widths.items():
            values = np.asarray(raw[name][field])
            if values.shape != (frames, width) or not np.isfinite(values).all():
                raise ValueError(f"Invalid source array {name}/{field}: {values.shape}")
    with np.load(soma_path, allow_pickle=False) as soma:
        if len(soma["poses"]) != frames or str(soma["object_id"].item()) != object_id:
            raise ValueError("MANO/SOMA frame count or object ID differs")
        for source_field, soma_field in (("rot", "object_rot"), ("trans", "object_trans")):
            if not np.array_equal(raw[object_id][source_field], soma[soma_field]):
                raise ValueError(f"MANO/SOMA object trajectory differs: {source_field}")
        soma_pose_shape = soma["poses"].shape
    print(f"GRAB comparison slot: {obj['number']:02d} {obj['grab_object']}")
    print(f"GraspXL object ID: {object_id}; size: {obj['size_variant']}")
    print(f"Frames: {frames}; original and converted frame indices correspond exactly")
    print(f"MANO: {mano_path}")
    print(f"SOMA: {soma_path}")
    print(f"Mesh: {mesh_path}")
    print(f"MANO hand pose: {raw['right_hand']['pose'].shape}; SOMA poses: {soma_pose_shape}")
    print("Object rotations/translations: exactly equal in MANO and SOMA files")
    print("MANO convention: right hand, use_pca=False, flat_hand_mean=False, zero betas")
    if args.viewer_root is None:
        print("To stage this motion for the official viewer, add --viewer-root PATH.")
        return

    viewer_root = args.viewer_root.resolve()
    viewer_script = package_file(viewer_root, "scripts/visualizer_mano.py")
    basename = f"{obj['object_key']}_{mano_path.parent.name}_{mano_path.stem}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", basename):
        raise ValueError(f"Unexpected characters in viewer basename: {basename}")
    destinations = [
        (mano_path, viewer_root / "data" / "GraspXL" / "recorded" / f"{basename}.npy"),
        (mesh_path, viewer_root / "data" / "GraspXL" / "object_mesh" / f"{basename}.obj"),
    ]
    for source, destination in destinations:
        try:
            destination.resolve().relative_to(viewer_root)
        except ValueError as exc:
            raise ValueError(f"Viewer destination leaves its repository: {destination}") from exc
        check_destination(source, destination)
    for source, destination in destinations:
        copy_unchanged(source, destination)
        print(f"Staged unchanged: {destination}")
    print(f"\nFrom the viewer repository directory ({viewer_script.parent.parent}), run:")
    print(f"python scripts/visualizer_mano.py --seq_name {basename} --obj_name {basename}")


if __name__ == "__main__":
    main()
