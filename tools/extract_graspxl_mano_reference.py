"""Extract original MANO files paired one-to-one with the matched SOMA package.

Run with the Windows Python 3.13 environment used by FilteredZipFile.
Only selected ZIP members are decompressed; split archives are read in place.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hand.convert_graspxl_to_soma import SplitReader, validate_sequence
from tools.extract_graspxl_matched_extras import FilteredZipFile


class BufferedSplitSource(SplitReader):
    def readinto(self, buffer):
        raw = self.read(len(buffer))
        buffer[:len(raw)] = raw
        return len(raw)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def package_path(root, relative):
    pure = PurePosixPath(relative)
    if pure.is_absolute() or '..' in pure.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe relative path: {relative}')
    path = root.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(relative)
    return path


def save_exact(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f'Existing file has different content: {path}')
    else:
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_bytes(raw)
        temporary.replace(path)
    if digest(path.read_bytes()) != digest(raw):
        raise ValueError(f'Written bytes differ: {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--soma-package', type=Path, default=ROOT.parent / 'GraspXL_SOMA_51Objects_GRABMatched')
    ap.add_argument('--input', type=Path, default=ROOT.parent / 'GraspXL DataSet')
    ap.add_argument('--output', type=Path, default=ROOT.parent / 'GraspXL_MANO_51Objects_GRABMatched')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    soma = args.soma_package.resolve(); out = args.output.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f'Destination exists; use --resume to verify and reuse identical files: {out}')
    if out == soma or out.is_relative_to(soma) or soma.is_relative_to(out):
        raise ValueError('MANO and SOMA packages must be separate sibling directories')
    if out.parent != soma.parent:
        raise ValueError('Use a sibling output directory for portable relative pairing')
    started = time.perf_counter()
    manifest = json.loads((soma / 'manifest.json').read_text(encoding='utf-8'))
    validation = json.loads((soma / 'reports/validation.json').read_text(encoding='utf-8'))
    source_index = json.loads((soma / 'metadata/sequence_index.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'converted_and_numerically_validated' or not validation['passed']:
        raise ValueError('Expected a validated SOMA package')
    assert len(source_index) == manifest['sequence_count'] == 749
    assert len(manifest['objects']) == manifest['object_count'] == 51
    objects = []
    for original in manifest['objects']:
        obj = {k:v for k,v in original.items() if k not in ('motion_paths', 'sequences_by_archive', 'source_mug_review')}
        obj['motion_paths'] = []
        obj['sequences_by_archive'] = {}
        mesh = package_path(soma, obj['mesh_path']).read_bytes()
        assert digest(mesh) == obj['mesh_sha256']
        save_exact(package_path(out, obj['mesh_path']), mesh)
        objects.append(obj)
    by_key = {obj['object_key']:obj for obj in objects}
    pairs = []
    def progress(status):
        write_json(out / 'reports/progress.json', dict(status=status,
            extracted_sequences=len(pairs), total_sequences=len(source_index),
            frames=sum(p['frames'] for p in pairs), elapsed_seconds=time.perf_counter()-started))
    progress('extracting')
    archive_reports = []
    for number in (1,2,3):
        name = f'mano_dataset_{number}'
        wanted = {r['source_member']:r for r in source_index if r['source_archive']==name}
        assert len(wanted) == sum(r['source_archive']==name for r in source_index)
        parts = [args.input / f'{name}.zip.part{i:02d}' for i in (0,1)]
        if not all(p.is_file() for p in parts): raise FileNotFoundError(parts)
        print(json.dumps(dict(event='index_archive', archive=name, expected=len(wanted))), flush=True)
        with io.BufferedReader(BufferedSplitSource(parts),buffer_size=1024*1024) as stream, \
                FilteredZipFile(stream, wanted.__contains__) as archive:
            assert set(archive.namelist()) == set(wanted), f'Missing selected members in {name}'
            for info in sorted(archive.infolist(), key=lambda x:x.header_offset):
                record = wanted[info.filename]
                assert f'{info.CRC:08x}' == record['source_crc32'], info.filename
                raw = archive.read(info)
                assert len(raw) == info.file_size
                data = np.load(io.BytesIO(raw), allow_pickle=True).item()
                uid, frames = validate_sequence(data)
                obj = by_key[record['object_key']]
                assert uid == obj['uid'] and frames == record['frames']
                assert PurePosixPath(info.filename).parts[-3] == obj['size_variant']
                target_relative = str(PurePosixPath(record['path']).with_suffix('.npy'))
                target = package_path(out, target_relative)
                save_exact(target, raw)
                soma_path = package_path(soma, record['path'])
                assert digest(soma_path.read_bytes()) == record['sha256']
                with np.load(soma_path, allow_pickle=False) as converted:
                    assert converted['poses'].shape[0] == frames
                    assert str(converted['object_id']) == uid
                    assert str(converted['source_sequence']) == f'{name}.zip:{info.filename}'
                    np.testing.assert_array_equal(converted['object_trans'], data[uid]['trans'])
                    np.testing.assert_array_equal(converted['object_rot'], data[uid]['rot'])
                pair = dict(object_key=obj['object_key'], grab_object=obj['grab_object'],
                    uid=uid, size_variant=obj['size_variant'], frames=frames,
                    mano_path=target_relative, soma_path=record['path'], mesh_path=obj['mesh_path'],
                    source_archive=name, source_member=info.filename, source_crc32=record['source_crc32'],
                    mano_sha256=digest(raw), soma_sha256=record['sha256'], bytes=len(raw),
                    original_bytes_preserved=True, object_transforms_match_soma=True)
                pairs.append(pair); obj['motion_paths'].append(target_relative)
                obj['sequences_by_archive'][name] = obj['sequences_by_archive'].get(name,0)+1
                if len(pairs)%50==0: progress('extracting')
            archive_reports.append(dict(archive=name, sequences=len(wanted),
                central_directory_entries_scanned=archive.members_scanned,
                source_parts=[dict(name=p.name,bytes=p.stat().st_size) for p in parts]))
        progress('extracting')
        print(json.dumps(dict(event='archive_extracted', archive=name, total=len(pairs))), flush=True)
    assert len(pairs)==len({p['mano_path'] for p in pairs})==len({p['soma_path'] for p in pairs})==749
    assert sum(p['frames'] for p in pairs)==manifest['frame_count']==116095
    for obj in objects:
        obj['motion_paths'].sort()
        assert len(obj['motion_paths']) == obj['sequence_count'] > 0
        write_json(package_path(out, str(PurePosixPath(obj['mesh_path']).parent / 'metadata.json')), obj)
    result = dict(schema_version=1, package_name=out.name, status='extracted_and_verified',
        object_count=51, unique_source_object_ids=len({o['uid'] for o in objects}),
        sequence_count=len(pairs), frame_count=sum(p['frames'] for p in pairs),
        soma_package_relative='../'+soma.name, format='Original GraspXL MANO .npy dictionaries',
        hand_type='right', unit='meters', fps=None,
        mano_convention=dict(use_pca=False, flat_hand_mean=False, betas='zero', pose_order='original MANO'),
        original_motion_bytes_preserved=True, objects=objects, source=manifest['source'])
    write_json(out/'manifest.json', result)
    write_json(out/'metadata/pairs.json', sorted(pairs,key=lambda p:p['mano_path']))
    write_json(out/'metadata/source_attribution.json', manifest['source'])
    write_json(out/'metadata/extraction_provenance.json', dict(extractor='tools/extract_graspxl_mano_reference.py',
        extractor_sha256=digest(Path(__file__).read_bytes()),
        source_soma_manifest_sha256=digest((soma/'manifest.json').read_bytes()), archives=archive_reports))
    shutil.copyfile(soma/'COMPARISON.md',out/'COMPARISON.md')
    shutil.copyfile(soma/'MOTION_COUNTS.md',out/'MOTION_COUNTS.md')
    (out/'scripts').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'tools/prepare_graspxl_mano_viewer.py',out/'scripts/prepare_viewer.py')
    shutil.copyfile(ROOT/'tools/GRASPXL_MANO_SOURCE_README.md',out/'README.md')
    report = dict(status='passed', objects=51, unique_source_ids=result['unique_source_object_ids'],
        sequences=len(pairs), frames=result['frame_count'],
        all_original_motion_bytes_preserved=True, all_selected_zip_crcs_passed=True,
        all_written_motion_sha256_verified=True, all_meshes_match_soma=True,
        one_to_one_soma_pairing=True, all_object_transforms_match_soma=True,
        raw_motion_bytes=sum(p['bytes'] for p in pairs),
        scope='Source extraction, finite MANO parameters, matching counts/mesh hashes and original object trajectories; physical contacts not revalidated.',
        elapsed_seconds=time.perf_counter()-started)
    write_json(out/'reports/validation.json',report)
    progress('extracted_and_verified')
    hashes={p.relative_to(out).as_posix():digest(p.read_bytes()) for p in out.rglob('*')
            if p.is_file() and p.name!='checksums.json'}
    write_json(out/'checksums.json',hashes)
    print(json.dumps(dict(event='verified',**report,
        package_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))),flush=True)


if __name__=='__main__':main()
