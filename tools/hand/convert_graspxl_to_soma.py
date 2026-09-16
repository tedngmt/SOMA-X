"""Stream GraspXL diverse-approach MANO archives into SOMA hand NPZ files."""

import argparse
import bisect
import io
import json
import os
import shutil
import time
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np


class SplitReader(io.RawIOBase):
    """Seekable, read-only concatenation of ZIP parts without a joined copy."""

    def __init__(self, parts):
        super().__init__()
        self.files = []
        self.offsets = [0]
        try:
            for path in parts:
                self.files.append(open(path, "rb"))
                self.offsets.append(self.offsets[-1] + Path(path).stat().st_size)
        except Exception:
            self.close()
            raise
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid whence")
        position = offset + (0, self.position, self.offsets[-1])[whence]
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.offsets[-1] - self.position)
        size = remaining if size is None or size < 0 else min(size, remaining)
        chunks = []
        while size:
            index = bisect.bisect_right(self.offsets, self.position) - 1
            self.files[index].seek(self.position - self.offsets[index])
            chunk = self.files[index].read(min(size, self.offsets[index + 1] - self.position))
            if not chunk:
                raise EOFError("Unexpected end of ZIP part")
            chunks.append(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def close(self):
        for file in self.files:
            file.close()
        super().close()


def archives(root):
    """Find only the three supported diverse-approach MANO datasets."""
    found = []
    for index in (1, 2, 3):
        name = f"mano_dataset_{index}"
        joined = root / f"{name}.zip"
        parts = sorted(root.glob(f"{name}.zip.part*"))
        if joined.exists():
            found.append((name, [joined]))
        elif parts:
            if [p.name for p in parts] != [f"{name}.zip.part{i:02d}" for i in range(len(parts))]:
                raise ValueError(f"Missing or misnumbered parts for {name}")
            found.append((name, parts))
    if not found:
        raise FileNotFoundError(f"No mano_dataset_[1-3].zip archives/parts in {root}")
    return found


def sequence_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    if path.suffix != ".npy" or not (path.name.startswith("mano_") or path.stem.isdigit()):
        return None
    # Some archives have an outer directory. Retain size/object/sequence only.
    if len(path.parts) < 3 or path.parts[-3] not in ("small", "medium", "large"):
        raise ValueError(f"Unexpected GraspXL layout: {name}")
    return Path(*path.parts[-3:])


def selected_members(archive, part_start=0, part_end=None):
    """Select complete file records inside a byte interval; omit boundary records.

    The next local header (or central directory) bounds each record, including
    optional ZIP data descriptors. Central-directory access still needs all parts.
    """
    entries = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
    for index, info in enumerate(entries):
        end = entries[index + 1].header_offset if index + 1 < len(entries) else archive.start_dir
        if info.header_offset < part_start or (part_end is not None and end > part_end):
            continue
        if sequence_path(info.filename) is not None:
            yield info


def validate_sequence(data):
    if not isinstance(data, dict) or "right_hand" not in data:
        raise ValueError("Expected sequence dictionary with right_hand")
    objects = [key for key in data if key != "right_hand"]
    if len(objects) != 1:
        raise ValueError("Expected exactly one object")
    frames = None
    for key in ("right_hand", objects[0]):
        fields = {"trans": 3, "rot": 3}
        if key == "right_hand":
            fields["pose"] = 45
        for field, width in fields.items():
            arr = np.asarray(data[key][field])
            if arr.ndim != 2 or arr.shape[1] != width or not np.isfinite(arr).all():
                raise ValueError(f"Invalid {key}/{field}: {arr.shape}")
            frames = arr.shape[0] if frames is None else frames
            if arr.shape[0] != frames or frames == 0:
                raise ValueError("Inconsistent or empty frame counts")
    return str(objects[0]), frames


def pack_rotations(rotations, storage="compact"):
    """Store float32 axis-angle rotations, or retain diagnostic matrices.

    Axis-angle has three components instead of nine matrix entries. Projection
    onto SO(3) removes the small numerical non-orthogonality from the fitter.
    No frame removal, float16 quantization, or PCA approximation is applied.
    """
    if storage == "diagnostic":
        return rotations
    if storage != "compact":
        raise ValueError(f"Unknown storage format: {storage}")
    from scipy.spatial.transform import Rotation

    shape = rotations.shape
    return Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().astype(
        np.float32
    ).reshape(*shape[:-2], 3)


class Converter:
    def __init__(self, args):
        model = args.data_root / "MANO" / "MANO_RIGHT.pkl"
        if not model.is_file():
            raise FileNotFoundError(f"Required licensed MANO model is missing: {model}")
        import smplx
        import torch
        from soma._smpl_family_loader import ensure_chumpy_compat
        from soma.fitting.pose_inversion import PoseInversion
        from soma.hand import SOMAHandLayer

        self.args = args
        self.torch = torch
        self.device = args.device
        ensure_chumpy_compat()
        self.mano = smplx.MANO(str(model), is_rhand=True, use_pca=False,
                               flat_hand_mean=False, create_transl=False).to(self.device)
        self.hand = SOMAHandLayer(data_root=str(args.data_root), hand_type="right",
                                  identity_model_type="mano", device=self.device).to(self.device)
        self.inv = PoseInversion(self.hand, low_lod=False)
        self.betas = torch.zeros(1, 10, device=self.device)
        with torch.no_grad():
            self.inv.prepare_identity(self.betas)

    def convert(self, data, destination, source, relative):
        from soma.io import save_soma_npz

        started = time.perf_counter()
        torch = self.torch
        object_id, frames = validate_sequence(data)
        storage = self.args.storage
        rotations, translations, errors = [], [], []
        hand = data["right_hand"]
        for start in range(0, frames, self.args.batch_size):
            stop = min(frames, start + self.args.batch_size)
            def tensor(field):
                return torch.as_tensor(hand[field][start:stop], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                # Match the official GraspXL visualizer, including the non-flat mean
                # and the MANO template wrist offset (trans is MANO translation).
                mano = self.mano(global_orient=tensor("rot"), hand_pose=tensor("pose"),
                                 betas=self.betas.expand(stop - start, -1))
                wrist = mano.joints[:, 0:1]
                target = self.hand.identity_model._to_soma_interp(mano.vertices - wrist)
            result = self.inv.fit(target, body_iters=0, finger_iters=0,
                                  full_iters=self.args.bcd_iters, lie_iters=self.args.lie_iters,
                                  batch_size=self.args.batch_size)
            if not all(torch.isfinite(result[key]).all() for key in
                       ("rotations", "root_translation", "per_vertex_error")):
                raise ValueError(f"Non-finite fitted parameters: {source}")
            rotations.append(result["rotations"].detach().cpu())
            translations.append((result["root_translation"] + wrist[:, 0] + tensor("trans")).detach().cpu())
            errors.append(result["per_vertex_error"].detach().mean(dim=-1).cpu())
        extra = {
            "source_dataset": np.array("GraspXL"), "source_sequence": np.array(source),
            "object_id": np.array(object_id),
            "object_mesh_archive": np.array("object_dataset.zip"),
            "object_mesh_member": np.array(f"{relative.parts[0]}/{object_id}/{object_id}.obj"),
            "object_trans": np.asarray(data[object_id]["trans"]),
            "object_rot": np.asarray(data[object_id]["rot"]),
            "mano_flat_hand_mean": np.bool_(False),
            "fit_mean_vertex_error_m": torch.cat(errors).numpy(),
            "conversion_settings": np.array(json.dumps({"bcd_iters": self.args.bcd_iters,
                                                         "lie_iters": self.args.lie_iters,
                                                         "storage": storage})),
        }
        if storage == "diagnostic":
            extra.update(mano_trans=np.asarray(hand["trans"]),
                         mano_rot=np.asarray(hand["rot"]), mano_pose=np.asarray(hand["pose"]))
        if "angle" in data[object_id]:
            extra["object_angle"] = np.asarray(data[object_id]["angle"])
        if self.args.fps is not None:
            extra["fps"] = np.float32(self.args.fps)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.npz")
        poses = pack_rotations(torch.cat(rotations).numpy(), storage)
        save_soma_npz(temporary, poses, torch.cat(translations),
                      joint_names=self.hand.rig_data["joint_names"], identity_model_type="mano",
                      identity_coeffs=self.betas, hand_type="right", keep_root=True,
                      unit="meters", extra_arrays=extra)
        os.replace(temporary, destination)
        return {"frames": frames, "mean_error_mm": float(torch.cat(errors).mean() * 1000),
                "seconds": time.perf_counter() - started, "bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--input-dir", type=Path, default=root.parent / "GraspXL DataSet")
    parser.add_argument("--output-dir", type=Path, default=root / "out/graspxl_soma_compact")
    parser.add_argument("--storage", choices=("compact", "diagnostic"), default="compact",
                        help="compact: float32 axis-angle without duplicate MANO motion (default); diagnostic: matrices plus original MANO motion")
    parser.add_argument("--data-root", type=Path, default=root / "assets")
    parser.add_argument("--inspect", action="store_true", help="Inspect archives and a sample without MANO assets")
    parser.add_argument("--archive", type=int, choices=(1, 2, 3), help="Process only this MANO archive")
    parser.add_argument("--part", type=int, help="Only complete records within this split part (0 = part00); requires --archive")
    parser.add_argument("--sample-count", type=int, help="Convert this many evenly spaced sequences from the selection")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Maximum frames per GPU fitting batch (default: 256; lower for smaller GPUs)")
    parser.add_argument("--bcd-iters", type=int, default=1)
    parser.add_argument("--lie-iters", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, help="Optional known source FPS; no guessed default")
    parser.add_argument("--resume", action="store_true", help="Skip completed outputs from this converter")
    args = parser.parse_args()
    if args.part is not None and (args.archive is None or args.part < 0):
        parser.error("--part requires --archive and a nonnegative index")
    if args.sample_count is not None and (args.sample_count < 1 or args.archive is None):
        parser.error("--sample-count must be positive and requires --archive")
    if args.batch_size < 1 or (args.max_sequences is not None and args.max_sequences < 1):
        parser.error("batch-size and max-sequences must be positive")
    if args.bcd_iters < 0 or args.lie_iters < 0 or (args.fps is not None and args.fps <= 0):
        parser.error("iterations must be nonnegative and fps positive")
    sources = archives(args.input_dir)
    if args.archive is not None:
        sources = [(name, parts) for name, parts in sources if name == f"mano_dataset_{args.archive}"]
        if not sources:
            parser.error("Selected archive was not found")
    converter = None if args.inspect else Converter(args)
    if not args.inspect:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for name, parts in sources:
        with SplitReader(parts) as stream, zipfile.ZipFile(stream) as archive:
            if args.part is not None:
                if len(parts) == 1 or args.part >= len(parts):
                    parser.error("Selected split part is unavailable (a joined ZIP cannot be part-filtered)")
                members = list(selected_members(archive, stream.offsets[args.part], stream.offsets[args.part + 1]))
            else:
                members = list(selected_members(archive))
            selected_count = len(members)
            print(json.dumps({"archive": name, "part": args.part, "selected_sequences": selected_count}), flush=True)
            if args.sample_count and members:
                indices = np.linspace(0, len(members) - 1, min(args.sample_count, len(members)), dtype=int)
                members = [members[index] for index in indices]
            count = 0
            sample = None
            for info in members:
                relative = sequence_path(info.filename)
                if relative is None:
                    continue
                count += 1
                if args.inspect and sample is not None:
                    continue
                destination = args.output_dir / name / relative.with_suffix(".npz")
                if not args.inspect and destination.exists():
                    if args.resume:
                        with np.load(destination, allow_pickle=False) as saved:
                            settings = json.loads(str(saved["conversion_settings"]))
                        expected = {"bcd_iters": args.bcd_iters, "lie_iters": args.lie_iters,
                                    "storage": args.storage}
                        settings.setdefault("storage", "diagnostic")
                        if settings != expected:
                            raise ValueError(f"Existing output settings differ: {destination}; choose a new output directory")
                        continue
                    raise FileExistsError(f"Output already exists: {destination}; use --resume")
                # GraspXL distributes pickled NumPy dictionaries. Load trusted downloads only.
                with archive.open(info) as file:
                    data = np.load(io.BytesIO(file.read()), allow_pickle=True).item()
                object_id, frames = validate_sequence(data)
                if object_id != relative.parent.name:
                    raise ValueError(f"Object ID does not match path: {info.filename}")
                if args.inspect:
                    sample = {"member": info.filename, "object_id": object_id, "frames": frames,
                              "fields": {k: {f: list(np.shape(v)) for f, v in d.items()} for k, d in data.items()}}
                else:
                    if shutil.disk_usage(args.output_dir).free < 1024 ** 3:
                        raise OSError("Less than 1 GiB free on the output drive; conversion stopped. "
                                      "Free space or choose another --output-dir, then resume.")
                    stats = converter.convert(data, destination, f"{name}.zip:{info.filename}", relative)
                    print(json.dumps({"output": str(destination), **stats}), flush=True)
                    completed += 1
                    if args.max_sequences and completed >= args.max_sequences:
                        return
            if args.inspect:
                print(json.dumps({"archive": name, "bytes": stream.offsets[-1], "sequences": count,
                                  "sample": sample}), flush=True)
            if count == 0:
                raise ValueError(f"No supported sequences in {name}")
    if not args.inspect:
        print(f"Converted {completed} sequences.", flush=True)


if __name__ == "__main__":
    main()
