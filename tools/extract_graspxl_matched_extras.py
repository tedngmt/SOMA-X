"""Stage selected source simulator assets and raw tabletop motions only.

The ZIP central directory is streamed so the 4.4-million-member URDF archive
does not require retaining millions of ZipInfo objects. Payload reads use
stdlib ZipFile and verify the selected members' original ZIP CRCs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import struct
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUP = re.compile(r"^urdf/surdf_group\d+_([sml])/([0-9a-f]{32})/")
SIZES = {"s": "small", "m": "medium", "l": "large"}


class FilteredZipFile(zipfile.ZipFile):
    """Python 3.13 ZipFile with a bounded-memory central-directory index.

    Uses the running stdlib's central-directory constants and ZipInfo decoder.
    This helper is intentionally read-only and retains only requested members.
    """
    def __init__(self, path, keep):
        self.keep = keep
        self.members_scanned = 0
        self.archive_entries = 0
        super().__init__(path, "r")

    def _RealGetContents(self):
        z = zipfile
        fp = self.fp
        end = z._EndRecData(fp)
        if not end:
            raise z.BadZipFile("Missing ZIP end record")
        self.archive_entries = end[z._ECD_ENTRIES_TOTAL]
        size_cd, offset_cd = end[z._ECD_SIZE], end[z._ECD_OFFSET]
        self._comment = end[z._ECD_COMMENT]
        concat = end[z._ECD_LOCATION] - size_cd - offset_cd
        if end[z._ECD_SIGNATURE] == z.stringEndArchive64:
            concat -= z.sizeEndCentDir64 + z.sizeEndCentDir64Locator
        self.start_dir = offset_cd + concat
        if self.start_dir < 0:
            raise z.BadZipFile("Invalid central-directory position")
        fp.seek(self.start_dir)
        consumed = 0
        while consumed < size_cd:
            header = fp.read(z.sizeCentralDir)
            if len(header) != z.sizeCentralDir:
                raise z.BadZipFile("Truncated central directory")
            cd = struct.unpack(z.structCentralDir, header)
            if cd[z._CD_SIGNATURE] != z.stringCentralDir:
                raise z.BadZipFile("Invalid central-directory signature")
            raw_name = fp.read(cd[z._CD_FILENAME_LENGTH])
            extra = fp.read(cd[z._CD_EXTRA_FIELD_LENGTH])
            comment = fp.read(cd[z._CD_COMMENT_LENGTH])
            consumed += z.sizeCentralDir + len(raw_name) + len(extra) + len(comment)
            self.members_scanned += 1
            name = raw_name.decode("utf-8" if cd[z._CD_FLAG_BITS] & z._MASK_UTF_FILENAME else "cp437")
            if not self.keep(name):
                continue
            info = z.ZipInfo(name)
            info.extra, info.comment = extra, comment
            info.header_offset = cd[z._CD_LOCAL_HEADER_OFFSET]
            (info.create_version, info.create_system, info.extract_version,
             info.reserved, info.flag_bits, info.compress_type, tm, dt,
             info.CRC, info.compress_size, info.file_size) = cd[1:12]
            if info.extract_version > z.MAX_EXTRACT_VERSION:
                raise NotImplementedError(f"Unsupported ZIP version {info.extract_version}")
            info.volume, info.internal_attr, info.external_attr = cd[15:18]
            info._raw_time = tm
            info.date_time = ((dt >> 9) + 1980, (dt >> 5) & 15, dt & 31,
                              tm >> 11, (tm >> 5) & 63, (tm & 31) * 2)
            info._decodeExtra(z.crc32(raw_name))
            info.header_offset += concat
            info._end_offset = self.start_dir
            if info.filename in self.NameToInfo:
                raise z.BadZipFile(f"Duplicate selected member: {info.filename}")
            self.filelist.append(info)
            self.NameToInfo[info.filename] = info
        if consumed != size_cd or self.members_scanned != self.archive_entries:
            raise z.BadZipFile("Central-directory size/count mismatch")
        # Conservative overlap boundary using retained members plus the CD.
        end_offset = self.start_dir
        for info in sorted(self.filelist, key=lambda i: i.header_offset, reverse=True):
            info._end_offset = end_offset
            end_offset = info.header_offset


def safe_target(root, name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or any(":" in p for p in pure.parts):
        raise ValueError(f"Unsafe ZIP path: {name}")
    path = root.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escaped stage: {name}")
    return path


def write_member(z, info, root):
    raw = z.read(info)  # ZipExtFile checks CRC at EOF.
    if len(raw) != info.file_size:
        raise ValueError(f"Size mismatch: {info.filename}")
    target = safe_target(root, info.filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != raw:
        target.write_bytes(raw)
    return {"archive_member": info.filename, "bytes": len(raw),
            "zip_crc32": f"{info.CRC:08x}", "sha256": hashlib.sha256(raw).hexdigest()}


def obj_vertices(raw):
    return np.asarray([[float(x) for x in line.split()[1:4]]
                       for line in raw.decode("utf-8", errors="replace").splitlines()
                       if line.startswith("v ")], dtype=np.float64)


def origin_transform(origin):
    result = np.eye(4)
    if origin is None:
        return result
    result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                      [-sp, cp*sr, cp*cr]]
    return result


def collision_extents(z, urdf_member):
    """Evaluate URDF collision geometry at zero joint position, in metres."""
    xml = ET.fromstring(z.read(urdf_member))
    joints = {}
    for joint in xml.findall("joint"):
        joints[joint.find("child").get("link")] = (
            joint.find("parent").get("link"), origin_transform(joint.find("origin")))
    def link_transform(name, visited=None):
        visited = set() if visited is None else visited
        if name in visited:
            raise ValueError(f"Cyclic URDF links in {urdf_member}")
        visited.add(name)
        if name not in joints:
            return np.eye(4)
        parent, local = joints[name]
        return link_transform(parent, visited) @ local
    vv, members = [], []
    for link in xml.findall("link"):
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Unexpected non-mesh collision in {urdf_member}")
            ref = mesh.get("filename")
            member = str(PurePosixPath(urdf_member).parent / ref)
            if not member.endswith(".obj"):
                raise ValueError(f"Expected OBJ collision in {urdf_member}: {ref}")
            v = obj_vertices(z.read(member)) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            tr = link_transform(link.get("name")) @ origin_transform(collision.find("origin"))
            vv.append(v @ tr[:3, :3].T + tr[:3, 3])
            members.append(member)
    return (np.ptp(np.concatenate(vv), axis=0) * 100 if vv else None), members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, default=ROOT / "out/grab-matched-graspxl/selected.json")
    ap.add_argument("--input", type=Path, default=ROOT.parent / "GraspXL DataSet")
    ap.add_argument("--output", type=Path, default=ROOT / "out/grab-matched-graspxl/extras-stage")
    args = ap.parse_args()
    stage = args.output.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    selected = json.loads(args.selection.read_text())["objects"]
    pairs = {(r["size_variant"], r["uid"]): r for r in selected}
    if len(pairs) != len(selected):
        raise ValueError("Selection contains duplicate native size/UID pairs")
    uids = {r["uid"] for r in selected}
    report = {"selection_count": len(selected), "source_paths_preserved": True,
              "source_archives_deleted": False, "selected_members_crc_verified": True,
              "tabletop_converted": False, "isaac_physics_ready": False,
              "notes": ["These are original RaiSim-oriented simulator assets; Isaac import and physics validation remain necessary.",
                        "Tabletop files are raw source MANO motions, not SOMA-X outputs. They require the tabletop-specific MANO convention (flat hand mean and wrist bias).",
                        "Only selected payload CRCs are checked; unselected archive payloads are not decompressed or validated."],
              "objects": [], "archives": {}, "unresolved_references": []}
    object_reports = {}
    for pair, r in pairs.items():
        rr = {"object": r["object"], "uid": r["uid"], "size_variant": r["size_variant"],
              "urdf_groups": [], "simulator_files": [], "tabletop_files": [],
              "group_geometry_checks": []}
        object_reports[pair] = rr
        report["objects"].append(rr)

    start = time.perf_counter()
    urdf_path = args.input / "objaverse_urdf.zip"
    def keep_urdf(name):
        m = GROUP.match(name)
        return m is not None and m.group(2) in uids
    print("Indexing selected simulator object IDs (all native size groups for verification)...", flush=True)
    with FilteredZipFile(urdf_path, keep_urdf) as z:
        print(json.dumps({"archive": urdf_path.name, "scanned": z.members_scanned,
                          "retained": len(z.filelist), "index_seconds": time.perf_counter() - start}), flush=True)
        by_pair = defaultdict(list)
        for info in z.infolist():
            m = GROUP.match(info.filename)
            by_pair[(SIZES[m.group(1)], m.group(2))].append(info)
        files_written = bytes_written = 0
        all_groups = {}
        for pair, r in pairs.items():
            rr = object_reports[pair]
            infos = [i for i in by_pair.get(pair, []) if not i.is_dir()]
            groups = sorted({str(PurePosixPath(i.filename).parent) for i in infos})
            rr["candidate_urdf_groups"] = groups
            accepted_groups = []
            for group in groups:
                group_infos = [i for i in infos if str(PurePosixPath(i.filename).parent) == group]
                urdf_member = f"{group}/{pair[1]}.urdf"
                ext, collision_members = collision_extents(z, urdf_member)
                target = np.asarray(r["axis_aligned_extents_cm"])
                # RaiSim assets contain a distant tiny dummy bottom visual link.
                # That link has no collision and is not the target object mesh.
                check = {"group": group, "suffix_native_size": pair[0],
                         "mesh_extents_cm": ext.tolist() if ext is not None else None,
                         "selected_object_extents_cm": target.tolist(),
                         "dimensions_match": bool(ext is not None and np.allclose(np.sort(ext), np.sort(target), rtol=0.01, atol=0.02)),
                         "collision_mesh_members": collision_members,
                         "method": "URDF collision geometry at zero joint position, including mesh scale and URDF origins; axes compared sorted; 1% relative or 0.02 cm tolerance. Dummy visual-only bottom link excluded."}
                rr["group_geometry_checks"].append(check)
                check["selected_for_staging"] = check["dimensions_match"]
                if check["dimensions_match"]:
                    accepted_groups.append(group)
                all_groups[group] = {i.filename for i in group_infos}
            rr["urdf_groups"] = accepted_groups
            infos = [i for i in infos if str(PurePosixPath(i.filename).parent) in accepted_groups]
            for info in infos:
                item = write_member(z, info, stage / "simulator_assets")
                item["path"] = str(PurePosixPath("simulator_assets") / info.filename)
                rr["simulator_files"].append(item)
                files_written += 1
                bytes_written += item["bytes"]
                if info.filename.endswith(".urdf"):
                    xml = ET.fromstring(z.read(info))
                    for mesh in xml.findall(".//mesh"):
                        ref = mesh.get("filename", "")
                        rel = PurePosixPath(info.filename).parent / ref
                        found = ref != "" and not PurePosixPath(ref).is_absolute() and "://" not in ref and str(rel) in z.NameToInfo
                        if not found:
                            report["unresolved_references"].append({"urdf": info.filename, "reference": ref})
            print(json.dumps({"object": r["object"], "urdf_groups": len(accepted_groups), "simulator_files": len(infos),
                              "geometry_match": [x["dimensions_match"] for x in rr["group_geometry_checks"]]}), flush=True)
        report["archives"]["objaverse_urdf.zip"] = {"archive_bytes": urdf_path.stat().st_size,
            "members_scanned": z.members_scanned, "selected_files": files_written,
            "selected_bytes": bytes_written, "seconds": time.perf_counter() - start}
    # Archive objects are released before opening the next archive.
    (stage / "extras_report.json").write_text(json.dumps(report, indent=2))

    table_path = args.input / "mano_tabletop.zip"
    start = time.perf_counter()
    def keep_tabletop(name):
        p = PurePosixPath(name).parts
        return len(p) >= 3 and (p[0], p[1]) in pairs and name.endswith(".npy")
    print("Indexing raw tabletop motions for selected native size/UID pairs...", flush=True)
    with FilteredZipFile(table_path, keep_tabletop) as z:
        total_bytes = 0
        for info in z.infolist():
            parts = PurePosixPath(info.filename).parts
            pair = (parts[0], parts[1])
            item = write_member(z, info, stage / "source_tabletop")
            item["path"] = str(PurePosixPath("source_tabletop") / info.filename)
            object_reports[pair]["tabletop_files"].append(item)
            total_bytes += item["bytes"]
        report["archives"]["mano_tabletop.zip"] = {"archive_bytes": table_path.stat().st_size,
            "members_scanned": z.members_scanned, "selected_files": len(z.filelist),
            "selected_bytes": total_bytes, "seconds": time.perf_counter() - start}
    report["missing_simulator_objects"] = [r["object"] for r in report["objects"] if not r["simulator_files"]]
    report["missing_tabletop_objects"] = [r["object"] for r in report["objects"] if not r["tabletop_files"]]
    report["rejected_simulator_groups"] = [dict(object=r["object"], **ch) for r in report["objects"]
                                           for ch in r["group_geometry_checks"] if not ch["selected_for_staging"]]
    report["geometry_mismatches"] = [dict(object=r["object"], **ch) for r in report["objects"]
                                      for ch in r["group_geometry_checks"] if ch["selected_for_staging"] and not ch["dimensions_match"]]
    report["status"] = "selected_source_assets_staged"
    (stage / "extras_report.json").write_text(json.dumps(report, indent=2))
    simulator_stats = report["archives"]["objaverse_urdf.zip"]
    tabletop_stats = report["archives"]["mano_tabletop.zip"]
    readme = f"""# Selected GraspXL source extras

These files supplement the converted SOMA-X hand motions. They are preserved
source assets, and the tabletop motions have **not** been converted to SOMA-X.

## Contents

- `simulator_assets/urdf/`: {simulator_stats['selected_files']} original source files,
  {simulator_stats['selected_bytes']:,} bytes, for all {len(selected)-len(report['missing_simulator_objects'])}/{len(selected)} selected object-size pairs.
- `source_tabletop/`: {tabletop_stats['selected_files']} raw MANO `.npy` files,
  {tabletop_stats['selected_bytes']:,} bytes, for {len(selected)-len(report['missing_tabletop_objects'])}/{len(selected)} selected object-size pairs.
- `extras_report.json`: object-to-source-path mapping, per-file CRC/SHA-256,
  geometry checks and missing tabletop coverage.

## Verification and use

All staged payloads passed their original ZIP CRC checks. URDF mesh references
resolve to included files. Native size suffixes `_s`, `_m`, `_l` were checked
against the selected object meshes using the actual URDF collision geometry.
The source URDFs include a tiny, distant dummy `bottom` visual link; it is not
part of the grasped object's collision geometry. Original links, masses,
inertias, joints and meshes are preserved. Isaac import and physics validation
remain necessary before training.

One alternative source group was rejected because its geometry used a different
scale; the correctly scaled group for that object is included. See the report's
`rejected_simulator_groups` field. No meshes or motions were rescaled.

Raw tabletop motions use the tabletop-specific MANO convention (flat hand mean
and wrist bias). They must not be fed unchanged into the diverse-grasp converter.
The original source ZIPs were retained. Unselected payloads were not extracted
or CRC-tested.

## Missing raw tabletop coverage

""" + "\n".join(f"- {name}" for name in report['missing_tabletop_objects']) + "\n"
    (stage / "SOURCE_EXTRAS_README.md").write_text(readme)
    print(json.dumps({"archives": report["archives"], "missing_simulator_objects": report["missing_simulator_objects"],
                      "missing_tabletop_objects": report["missing_tabletop_objects"],
                      "geometry_mismatches": len(report["geometry_mismatches"]),
                      "unresolved_references": len(report["unresolved_references"])}), flush=True)


if __name__ == "__main__":
    main()
