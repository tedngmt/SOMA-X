"""Convert all motions for the 51 GraspXL object-size pairs matched to GRAB."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hand.convert_graspxl_to_soma import Converter, SplitReader, sequence_path, validate_sequence


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, default=ROOT.parent / 'GraspXL DataSet')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new empty destination: {output}')
    started = time.perf_counter()
    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text())
    records = []
    by_pair = {}
    output.mkdir(parents=True)
    for selected in selection['objects']:
        key = f'{selected["number"]:02d}_{selected["grab_object"]}_{selected["uid"]}'
        mesh_path = f'objects/{key}/mesh.obj'
        source_mesh = selection_path.parent / selected['local_mesh']
        target_mesh = output / mesh_path
        target_mesh.parent.mkdir(parents=True)
        shutil.copyfile(source_mesh, target_mesh)
        record = {k: selected[k] for k in ['number', 'object', 'uid', 'size_variant', 'lvis_category',
                  'difficulty', 'challenge', 'axis_aligned_extents_cm', 'vertices', 'faces', 'mesh_member', 'grab_object', 'match_type', 'match_notes', 'grab_extents_cm']}
        record.update(object_key=key, mesh_path=mesh_path, mesh_sha256=sha256(target_mesh),
                      mesh_unit='meters', motion_paths=[], sequences_by_archive={})
        if 'mug_review' in selected:
            record['source_mug_review'] = selected['mug_review']
        records.append(record)
        by_pair[(selected['size_variant'], selected['uid'])] = record
    assert len(records) == len(by_pair) == 51
    config = argparse.Namespace(data_root=ROOT / 'assets', device=args.device, batch_size=256,
                                bcd_iters=1, lie_iters=3, storage='compact', fps=None)
    converter = Converter(config)
    sequence_records = []
    archive_reports = []
    for number in (1, 2, 3):
        archive_name = f'mano_dataset_{number}'
        parts = [args.input_dir / f'{archive_name}.zip.part{i:02d}' for i in (0, 1)]
        if not all(path.is_file() for path in parts):
            raise FileNotFoundError(f'Both split parts required: {parts}')
        print(json.dumps(dict(event='indexing_archive', archive=archive_name)), flush=True)
        archive_start = time.perf_counter()
        with SplitReader(parts) as stream, zipfile.ZipFile(stream) as archive:
            matches = []
            for info in archive.infolist():
                if not info.filename.endswith('.npy'):
                    continue
                relative = sequence_path(info.filename)
                if relative is not None and (relative.parts[0], relative.parent.name) in by_pair:
                    matches.append((info, relative))
            matches.sort(key=lambda item: item[0].filename)
            print(json.dumps(dict(event='selected_archive_motions', archive=archive_name,
                                  sequences=len(matches))), flush=True)
            per_archive = Counter()
            for index, (info, relative) in enumerate(matches, 1):
                record = by_pair[(relative.parts[0], relative.parent.name)]
                data = np.load(io.BytesIO(archive.read(info)), allow_pickle=True).item()
                uid, frames = validate_sequence(data)
                assert uid == record['uid']
                motion_path = f'motions/{record["object_key"]}/{archive_name}/{relative.stem}.npz'
                destination = output / motion_path
                stats = converter.convert(data, destination, f'{archive_name}.zip:{info.filename}', relative)
                # Add a portable package reference and the actual full source ZIP member.
                with np.load(destination, allow_pickle=False) as packed:
                    arrays = {key: packed[key] for key in packed.files}
                np.testing.assert_array_equal(arrays['object_trans'], data[uid]['trans'])
                np.testing.assert_array_equal(arrays['object_rot'], data[uid]['rot'])
                arrays['object_mesh_member'] = np.array(record['mesh_member'])
                arrays['package_mesh_path'] = np.array(record['mesh_path'])
                arrays['object_size_variant'] = np.array(record['size_variant'])
                arrays['grab_counterpart'] = np.array(record['grab_object'])
                arrays['object_match_type'] = np.array(record['match_type'])
                temporary = destination.with_suffix('.partial.npz')
                np.savez_compressed(temporary, **arrays)
                temporary.replace(destination)
                errors = arrays['fit_mean_vertex_error_m']
                record['motion_paths'].append(motion_path)
                per_archive[record['object_key']] += 1
                sequence_records.append(dict(path=motion_path, object_key=record['object_key'],
                      source_archive=archive_name, source_member=info.filename,
                      source_crc32=f'{info.CRC:08x}', frames=frames,
                      mean_fit_error_mm=float(np.mean(errors) * 1000),
                      max_frame_mean_fit_error_mm=float(np.max(errors) * 1000),
                      bytes=destination.stat().st_size, sha256=sha256(destination),
                      object_transforms_exactly_preserved=True))
                if index % 20 == 0 or index == len(matches):
                    write_json(output / 'reports/progress.json', dict(status='converting', archive=archive_name, archive_done=index, archive_total=len(matches), converted_sequences=len(sequence_records)))
                    print(json.dumps(dict(event='progress', archive=archive_name, done=index,
                                          total=len(matches), converted_total=len(sequence_records))), flush=True)
            archive_reports.append(dict(archive=archive_name, selected_sequences=len(matches),
                                        seconds=time.perf_counter()-archive_start,
                                        parts=[dict(name=p.name, bytes=p.stat().st_size) for p in parts]))
            for record in records:
                record['sequences_by_archive'][archive_name] = per_archive[record['object_key']]
        del archive, matches
        gc.collect()
    for record in records:
        if not record['motion_paths']:
            raise ValueError(f'No motions for {record["object_key"]}')
        record['sequence_count'] = len(record['motion_paths'])
        write_json(output / f'objects/{record["object_key"]}/metadata.json', record)
    source = dict(dataset_url='https://huggingface.co/datasets/ethHuiZhang/GraspXL',
                  dataset_license='CC-BY-NC-4.0',
                  citation='Zhang et al., GraspXL: Generating Grasping Motions for Diverse Objects at Scale, ECCV 2024.',
                  mano_model='MANO v1.2 right hand; separate license; model not bundled',
                  mano_right_sha256=sha256(ROOT / 'assets/MANO/MANO_RIGHT.pkl'))
    manifest = dict(schema_version=1, package_name=output.name,
                    created_utc=datetime.now(timezone.utc).isoformat(), status='converted_pending_validation',
                    object_count=len(records), sequence_count=len(sequence_records),
                    frame_count=sum(r['frames'] for r in sequence_records),
                    scope='All available diverse MANO archive 1/2/3 motions for the 51 selected object-size pairs matched to GRAB.',
                    hand_type='right', format='SOMAHandLayer compact NPZ', unit='meters', fps=None,
                    coordinates='Original GraspXL world and object-local coordinates; no axis changes or rescaling.',
                    full_body_motion=False, physics_validated=False,
                    settings=dict(batch_size=256, bcd_iters=1, lie_iters=3, storage='compact',
                                  mano_flat_hand_mean=False, mano_betas='zero', device=args.device),
                    source=source, objects=records)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'metadata/sequence_index.json', sequence_records)
    write_json(output / 'metadata/source_selection.json', selection)
    write_json(output / 'metadata/source_attribution.json', source)
    (output / 'previews').mkdir()
    for src in selection_path.parent.glob('selected-page-*.png'):
        shutil.copyfile(src, output / 'previews' / src.name)
    shutil.copyfile(selection_path.parent / 'COMPARISON.md', output / 'COMPARISON.md')
    report = dict(objects=len(records), sequences=len(sequence_records), frames=manifest['frame_count'],
                  archives=archive_reports, elapsed_seconds=time.perf_counter()-started,
                  npz_bytes=sum(r['bytes'] for r in sequence_records),
                  mesh_bytes=sum((output / r['mesh_path']).stat().st_size for r in records),
                  source_object_transforms_exactly_preserved=True)
    write_json(output / 'reports/conversion.json', report)
    print(json.dumps(dict(event='conversion_complete', **report)), flush=True)
    sys.path.insert(0, str(ROOT / 'out/household-pilot'))
    from validate_package import validate
    validation = validate(output, ROOT / 'assets', args.device)
    write_json(output / 'reports/validation.json', validation)
    if not validation['passed']:
        raise RuntimeError(validation['errors'])
    assert validation['valid_motion_count'] == manifest['sequence_count']
    assert validation['frame_count'] == manifest['frame_count']
    manifest['status'] = 'converted_and_numerically_validated'
    write_json(output / 'reports/progress.json', dict(status=manifest['status'], objects=len(records), sequences=len(sequence_records), frames=manifest['frame_count']))
    manifest['unique_source_object_ids'] = len({r['uid'] for r in records})
    manifest['comparison_notes'] = 'COMPARISON.md'
    manifest['match_types'] = dict(Counter(r['match_type'] for r in records))
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'metadata/comparison_groups.json', dict(same_type=[r['grab_object'] for r in records if r['match_type']=='same_type'], proxy=[r['grab_object'] for r in records if r['match_type']!='same_type'], split_group_by_source_uid={uid:[r['grab_object'] for r in records if r['uid']==uid] for uid in sorted({r['uid'] for r in records})}))
    write_json(output / 'metadata/build_provenance.json', dict(builder='tools/graspxl_matched_package.py', validator='out/household-pilot/validate_package.py', builder_sha256=sha256(Path(__file__)), converter_sha256=sha256(ROOT / 'tools/hand/convert_graspxl_to_soma.py'), selection_sha256=sha256(selection_path)))
    (output / 'scripts').mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'out/household-pilot/package_loader.py', output / 'scripts/load_sequence.py')
    base = (ROOT / 'out/household-pilot/PACKAGE_README.md').read_text(encoding='utf-8')
    base = base.replace('20 household objects', '51 GRAB-matched object entries').replace('the 20 selected', 'the 51 selected')
    base = base.replace('Object shortlist and SOURCE MANO mug preview', 'Contact sheets of the selected source meshes')
    para_start = base.index('The handled mug is object 1.')
    para_end = base.index('## Folder layout', para_start)
    base = base[:para_start] + ('Read **COMPARISON.md** before comparing datasets. Entries are paired by object type or a documented proxy; dimensions and motion content differ. GraspXL provides floating right-hand grasps, while GRAB provides whole-body activities. Source FPS is unknown. No action or duration equivalence is claimed.\n\nThe selected mug has an open handle. Its previously inspected source motions grasped the body or rim; a handle-specific grasp has not been established.\n\n' + base[para_end:])
    summary = f"Objects/size entries: {len(records)}; unique source meshes: {manifest['unique_source_object_ids']}; sequences: {len(sequence_records)}; frames: {manifest['frame_count']}. All numerical checks passed.\n\n"
    (output / 'README.md').write_text(base.replace('## Folder layout', summary + '## Folder layout'), encoding='utf-8')
    write_json(output / 'checksums.json', {p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps(dict(event='validated', objects=len(records), sequences=len(sequence_records), frames=manifest['frame_count'], fit_error=validation['fit_error'])),flush=True)


if __name__ == '__main__':
    main()
