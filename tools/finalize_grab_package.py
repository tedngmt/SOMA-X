"""Validate every converted GRAB clip and finalize its package manifest."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def verify_source_and_replay(package, inventory, source_root, assets):
    """Independently reload one complete clip per subject, replay three frames."""
    import torch
    import smplx
    from soma.fitting.pose_inversion import PoseInversion
    from tools.replay_grab_soma import load_personalized_layer, replay

    torch.set_num_threads(4)
    records = []
    for subject in inventory['subjects']:
        candidates = [r for r in inventory['sequences'] if r['subject'] == subject]
        record = next((r for r in candidates if r['object'] == 'mug' and
                       r['action'] == 'drink'), candidates[0])
        relative = f'motions/{subject}/{Path(record["member"]).name}'
        with np.load(package / relative, allow_pickle=False) as data:
            motion = dict(data)
        with zipfile.ZipFile(source_root / record['archive']) as archive:
            with np.load(io.BytesIO(archive.read(record['member'])), allow_pickle=True) as data:
                original = dict(data)
        body = original['body'].item()['params']
        for name in ('object', 'table'):
            params = original[name].item()['params']
            np.testing.assert_array_equal(motion[f'{name}_trans'], params['transl'])
            np.testing.assert_array_equal(motion[f'{name}_rot'], params['global_orient'])
        np.testing.assert_array_equal(motion['source_expression'], body['expression'])
        with np.load(package / 'contacts' / subject / Path(record['member']).name,
                     allow_pickle=False) as contact:
            source_contact = original['contact'].item()
            np.testing.assert_array_equal(contact['body_contact'], source_contact['body'])
            np.testing.assert_array_equal(contact['object_contact'], source_contact['object'])
            np.testing.assert_array_equal(contact['threshold'], source_contact['threshold'])
        indices = np.array([0, record['frames'] // 2, record['frames'] - 1])
        layer = load_personalized_layer(package, motion, assets, 'cpu')
        result = replay(layer, motion, indices)
        for name in ('vertices', 'joints', 'transforms'):
            assert torch.isfinite(result[name]).all(), (relative, name)
        gender = str(motion['gender'])
        model = smplx.SMPLX(str(assets / 'SMPLX' / f'SMPLX_{gender.upper()}.npz'),
                           gender=gender, use_pca=True, num_pca_comps=24,
                           flat_hand_mean=False, batch_size=1,
                           v_template=layer.identity_model.identity_model.v_template)
        fields = ('global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
                  'jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'transl')
        params = {k: torch.as_tensor(body[k][indices], dtype=torch.float32) for k in fields}
        with torch.no_grad():
            source_body = model(**params, betas=torch.zeros(len(indices), 10))
            reference = PoseInversion(layer, low_lod=False).transfer_to_soma(source_body.vertices)
        errors = (result['vertices'] - reference).norm(dim=-1).mean(dim=-1) * 1000
        conversion_report = json.loads((package / 'reports/sequences' / subject /
                                       (Path(record['member']).stem + '.json')).read_text())
        previous = {r['frame']: r['replay_mean_error_mm'] for r in conversion_report['replay_checks']}
        deltas = [abs(float(error) - previous[int(frame)]) for frame, error in zip(indices, errors)]
        assert max(deltas) < 0.1, (relative, 'CPU replay differs from conversion replay', deltas)
        records.append(dict(subject=subject, motion=relative, source_arrays_exact=True,
                            frame_indices=indices.tolist(), replay_mean_error_mm=errors.tolist(),
                            max_difference_from_conversion_replay_mm=max(deltas)))
        sample_path = package / 'reports/replay_samples' / f'{subject}.npz'
        sample_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(sample_path, vertices=result['vertices'].detach().numpy(),
                            reference_vertices=reference.detach().numpy(),
                            faces=layer.faces.cpu().numpy(), frame_indices=indices,
                            motion_path=np.array(relative))
        print(json.dumps(dict(event='source_and_cpu_replay_validated', **records[-1])), flush=True)
        del model, layer, result, reference, original, motion
    summary = dict(status='passed', sampled_sequences=len(records), subjects=len(records),
                   replayed_frames=sum(len(r['frame_indices']) for r in records),
                   scope='Exact full-clip object/table transforms, contacts and expressions for one clip per subject; independent CPU replay of first/middle/last frames',
                   records=records)
    write_json(package / 'reports/source_and_replay_validation.json', summary)
    return summary


def finalize(package, source_root=None, assets=None, replay_checks=False):
    inventory = json.loads((ROOT / 'out/grab-preparation/inventory.json').read_text())
    report = json.loads((package / 'reports/conversion.json').read_text())
    assert not report['pilot']
    assert report['sequences'] == inventory['sequence_count']
    assert report['frames'] == inventory['total_frames']
    results = {r['output']: r for r in report['results']}
    assert len(results) == inventory['sequence_count']
    geometry = json.loads((ROOT / 'out/grab-preparation/geometry.json').read_text())
    counts = Counter()
    subject_counts = Counter()
    frames = 0
    error_sum = 0.0
    motions = []
    hashes = {}
    all_errors = []
    replay_errors = []
    expected_motions = set()
    expected_contacts = set()
    rigs = {}
    for subject, info in inventory['subjects'].items():
        folder = package / 'subjects' / subject
        with np.load(folder / 'soma_rig.npz', allow_pickle=False) as rig:
            rigs[subject] = rig['joint_names']
            assert rig['rest_vertices'].shape == (1, 4505, 3), subject
            assert rig['joint_names'].shape == (78,), subject
            assert rig['skinning_weights'].shape == (4505, 78), subject
            assert rig['bind_transforms_world'].shape == (1, 78, 4, 4), subject
            assert rig['root_joint_idx'] == 1, subject
            for key in rig.files:
                if np.issubdtype(rig[key].dtype, np.number):
                    assert np.isfinite(rig[key]).all(), (subject, key)
        template = ROOT / 'out/grab-preparation/subjects' / info['gender'] / f'{subject}.ply'
        assert template.read_bytes() == (folder / 'smplx_template.ply').read_bytes(), subject
    for record in inventory['sequences']:
        relative = f'motions/{record["subject"]}/{Path(record["member"]).name}'
        path = package / relative
        expected_motions.add(relative)
        with np.load(path, allow_pickle=False) as data:
            n = record['frames']
            assert data['poses'].shape == (n, 78, 3), relative
            assert data['transl'].shape == (n, 3), relative
            assert float(data['fps']) == 120, relative
            assert str(data['object_id']) == record['object'], relative
            assert str(data['subject_id']) == record['subject'], relative
            assert str(data['source_sequence']) == record['archive'] + ':' + record['member'], relative
            assert str(data['gender']) == inventory['subjects'][record['subject']]['gender'], relative
            assert str(data['rotation_repr']) == 'rotvec' and str(data['unit']) == 'meters', relative
            assert bool(data['keep_root']) and bool(data['absolute_pose']), relative
            assert bool(data['personalized_identity']), relative
            assert data['identity_coeffs'].shape == (1, 10) and not data['identity_coeffs'].any(), relative
            np.testing.assert_array_equal(data['joint_names'], rigs[record['subject']])
            for key in ('object_trans', 'object_rot', 'table_trans', 'table_rot'):
                assert data[key].shape == (n, 3), (relative, key, data[key].shape)
            assert data['source_expression'].shape == (n, 10), relative
            assert data['fit_mean_vertex_error_m'].shape == (n,), relative
            # Access every member: this also verifies every NPZ member's ZIP CRC.
            for key in data.files:
                value = data[key]
                if np.issubdtype(value.dtype, np.number):
                    assert np.isfinite(value).all(), (relative, key)
            for key in ('package_mesh_path', 'package_table_mesh_path',
                        'subject_rig_path', 'subject_template_path'):
                assert (package / str(data[key])).is_file(), (relative, key)
            errors = data['fit_mean_vertex_error_m']
            assert (errors >= 0).all(), relative
            error_sum += float(errors.sum(dtype=np.float64))
            all_errors.append(errors)
            clip_report = results[relative]
            assert clip_report['frames'] == n, relative
            assert abs(clip_report['mean_fit_error_mm'] - float(errors.mean()) * 1000) < 0.001, relative
            assert len(clip_report['replay_checks']) == 3, relative
            for check in clip_report['replay_checks']:
                assert 0 <= check['frame'] < n, relative
                assert np.isfinite(check['replay_mean_error_mm']), relative
                replay_errors.append(check['replay_mean_error_mm'])
        contact_path = package / 'contacts' / record['subject'] / path.name
        expected_contacts.add(contact_path.relative_to(package).as_posix())
        with np.load(contact_path, allow_pickle=False) as contacts:
            assert contacts['body_contact'].shape == (n, 10475), relative
            assert contacts['object_contact'].shape == (n, geometry[record['object']]['vertices']), relative
            assert np.issubdtype(contacts['body_contact'].dtype, np.integer), relative
            assert np.issubdtype(contacts['object_contact'].dtype, np.integer), relative
            assert np.isfinite(contacts['threshold']).all() and float(contacts['threshold']) >= 0, relative
            assert str(contacts['object_mesh_path']) == f'objects/{record["object"]}/mesh.ply', relative
            assert 'NOT SOMA' in str(contacts['body_topology']), relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[contact_path.relative_to(package).as_posix()] = hashlib.sha256(contact_path.read_bytes()).hexdigest()
        counts[record['object']] += 1
        subject_counts[record['subject']] += 1
        frames += n
        motions.append(dict(path=relative, subject=record['subject'],
                            object=record['object'], action=record['action'], frames=n, fps=120,
                            contacts=contact_path.relative_to(package).as_posix()))
        if len(motions) % 100 == 0:
            print(json.dumps(dict(event='validated_files', sequences=len(motions), total=len(inventory['sequences']))), flush=True)
    assert frames == inventory['total_frames']
    assert len(counts) == 51
    assert set(subject_counts) == set(inventory['subjects'])
    assert {p.relative_to(package).as_posix() for p in (package / 'motions').rglob('*.npz')} == expected_motions
    assert {p.relative_to(package).as_posix() for p in (package / 'contacts').rglob('*.npz')} == expected_contacts
    for obj in inventory['objects']:
        assert counts[obj['name']] == obj['sequence_count']
        source = ROOT / 'out/grab-preparation/objects' / (obj['name'] + '.ply')
        destination = package / 'objects' / obj['name'] / 'mesh.ply'
        assert source.read_bytes() == destination.read_bytes()
    assert (ROOT / 'out/grab-preparation/objects/table.ply').read_bytes() == (package / 'scene/table.ply').read_bytes()
    extra_validation = verify_source_and_replay(package, inventory, source_root, assets) if replay_checks else None
    all_errors = np.concatenate(all_errors) * 1000
    summary = dict(status='validated', sequences=len(motions), frames=frames,
                   objects=len(counts), subjects=len(inventory['subjects']), fps=120,
                   mean_fit_vertex_error_mm=1000 * error_sum / frames,
                   p95_frame_mean_fit_error_mm=float(np.percentile(all_errors, 95)),
                   max_frame_mean_fit_error_mm=float(all_errors.max()),
                   conversion_replay_checks=len(replay_errors),
                   mean_conversion_replay_error_mm=float(np.mean(replay_errors)),
                   max_conversion_replay_error_mm=float(np.max(replay_errors)),
                   subject_sequence_counts=dict(subject_counts),
                   object_sequence_counts=dict(counts),
                   independent_source_and_replay_checks=extra_validation is not None,
                   validation='Every motion and contact NPZ: CRC, finite numeric motion arrays, shapes, counts, identity, linked assets; all object/table meshes and personalized templates byte-identical; report consistency',
                   limitations='Approximate surface fit, not physical contact/penetration validation; body contacts retain SMPL-X topology, expression coefficients are source metadata.')
    write_json(package / 'reports/validation.json', summary)
    write_json(package / 'reports/data_sha256.json', hashes)
    manifest = json.loads((package / 'manifest.json').read_text())
    manifest.update({k: v for k, v in summary.items() if k not in ('objects', 'subjects')})
    # Keep indexes instead of overwriting them with integer counts from summary.
    manifest.update(objects=inventory['objects'], subjects=inventory['subjects'],
                    object_count=len(counts), subject_count=len(subject_counts), motions=motions,
                    contact_topology='Original SMPL-X body vertices; original object vertices',
                    replay='Use personalized SMPL-X templates and tools/replay_grab_soma.py',
                    validation_report='reports/validation.json')
    write_json(package / 'manifest.json', manifest)
    scripts = package / 'scripts'
    scripts.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'tools/replay_grab_soma.py', scripts / 'replay_grab_soma.py')
    shutil.copyfile(ROOT / 'tools/GRAB_PACKAGE_README.md', package / 'README.md')
    write_json(package / 'reports/progress.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'Grab Dataset')
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--replay-checks', action='store_true',
                        help='CPU replay and original-source equality checks for one full clip per subject')
    args = parser.parse_args()
    while args.wait and not (args.package / 'reports/conversion.json').is_file():
        time.sleep(15)
    finalize(args.package, args.source, args.assets, args.replay_checks)


if __name__ == '__main__':
    main()
