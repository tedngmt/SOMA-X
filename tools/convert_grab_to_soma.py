"""Convert GRAB's personalized SMPL-X bodies, preserving object/table motion."""
import argparse
import io
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
import smplx
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soma.body import SOMALayer
from soma.fitting.pose_inversion import PoseInversion
from soma.io import save_soma_npz


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


def json_save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def initialize_subject(subject, gender, prepared, assets, output, device, batch):
    source_template = prepared / 'subjects' / gender / f'{subject}.ply'
    mesh = trimesh.load(source_template, process=False, maintain_order=True)
    template = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    model = smplx.SMPLX(str(assets / 'SMPLX' / f'SMPLX_{gender.upper()}.npz'),
                        gender=gender, use_pca=True, num_pca_comps=24, flat_hand_mean=False,
                        v_template=template, batch_size=1).to(device)
    soma = SOMALayer(assets, lod='low', identity_model_type='smplx',
                     identity_model_kwargs={'gender':gender}, device=device, mode='warp')
    # GRAB provides personalized rest meshes, not a generic zero-beta person.
    soma.identity_model.identity_model.v_template = template
    inv = PoseInversion(soma, low_lod=False)
    with torch.no_grad():
        inv.prepare_identity(torch.zeros(1, 10, device=device))
    folder = output / 'subjects' / subject
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_template, folder / 'smplx_template.ply')
    view = soma.public_rig_view()
    def cpu(x): return x.detach().cpu().numpy()
    np.savez_compressed(folder / 'soma_rig.npz',
                        joint_names=np.array(view.joint_names),
                        joint_parent_ids=cpu(view.joint_parent_ids),
                        bind_transforms_world=cpu(view.bind_transforms_world),
                        rest_vertices=cpu(soma._cached_rest_shape),
                        skinning_weights=cpu(view.skinning_weights), faces=cpu(soma.faces),
                        t_pose_world=cpu(view.t_pose_world), unit=np.array('meters'),
                        root_joint_idx=np.int64(soma.root_joint_idx))
    json_save(folder / 'metadata.json', dict(subject=subject, gender=gender,
              identity='Personalized GRAB SMPL-X v_template with zero additional betas',
              rig='soma_rig.npz', template='smplx_template.ply', lod='low'))
    return model, soma, inv


def convert_sequence(data, model, soma, inv, args, target, source, subject):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    body = data['body'].item()['params']
    frames = int(data['n_frames'].item())
    assert int(data['n_comps'].item()) == 24
    fields = ['global_orient','body_pose','left_hand_pose','right_hand_pose',
              'jaw_pose','leye_pose','reye_pose','expression','transl']
    rotations, translations, errors = [], [], []
    sample_indices = {0, frames//2, frames-1}
    previews = []
    for start in range(0, frames, args.batch_size):
        stop = min(start + args.batch_size, frames)
        params = {k:torch.as_tensor(body[k][start:stop], dtype=torch.float32, device=args.device)
                  for k in fields}
        with torch.no_grad():
            source_body = model(**params, betas=torch.zeros(stop-start,10,device=args.device))
        fit = inv.fit(source_body.vertices, body_iters=2, finger_iters=1, full_iters=1,
                      lie_iters=3, batch_size=args.batch_size)
        if args.pilot:
            print(json.dumps(dict(event='pilot_batch',frames=stop,total=frames,
                  seconds=round(time.perf_counter()-started,2),
                  gpu_peak_allocated_mb=round(torch.cuda.max_memory_allocated()/1e6))),flush=True)
        for key in ['rotations','root_translation','per_vertex_error']:
            if not torch.isfinite(fit[key]).all(): raise ValueError(f'Non-finite {key}: {source}')
        rotations.append(fit['rotations'].detach().cpu().numpy())
        translations.append(fit['root_translation'].detach().cpu().numpy())
        errors.append(fit['per_vertex_error'].detach().mean(dim=-1).cpu().numpy())
        # Verify actual saved rotation representation on representative frames.
        for idx in sorted(sample_indices & set(range(start,stop))):
            j = idx-start
            rot = pack_rotations(fit['rotations'][j:j+1].detach().cpu().numpy())
            with torch.no_grad():
                replay = soma.pose(torch.tensor(rot[:,1:], device=args.device), pose2rot=True,
                          absolute_pose=True, transl=fit['root_translation'][j:j+1],
                          apply_correctives=False)['vertices']
                reference = inv.transfer_to_soma(source_body.vertices[j:j+1])
                err = (replay-reference).norm(dim=-1)
            previews.append(dict(frame=idx, replay_mean_error_mm=float(err.mean()*1000),
                                 stored_mean_error_mm=float(fit['per_vertex_error'][j].mean()*1000)))
    rotations = pack_rotations(np.concatenate(rotations))
    transl = np.concatenate(translations)
    per_frame = np.concatenate(errors)
    obj = data['object'].item()
    table = data['table'].item()
    name = str(data['obj_name'].item())
    extra = dict(source_dataset=np.array('GRAB'), source_sequence=np.array(source),
                 subject_id=np.array(subject), gender=data['gender'], motion_intent=data['motion_intent'],
                 fps=np.float32(data['framerate'].item()), object_id=np.array(name),
                 package_mesh_path=np.array(f'objects/{name}/mesh.ply'),
                 object_trans=np.asarray(obj['params']['transl']),
                 object_rot=np.asarray(obj['params']['global_orient']),
                 table_trans=np.asarray(table['params']['transl']),
                 table_rot=np.asarray(table['params']['global_orient']),
                 package_table_mesh_path=np.array('scene/table.ply'),
                 subject_rig_path=np.array(f'subjects/{subject}/soma_rig.npz'),
                 subject_template_path=np.array(f'subjects/{subject}/smplx_template.ply'),
                 identity_gender=data['gender'], personalized_identity=np.bool_(True),
                 fit_mean_vertex_error_m=per_frame, source_expression=np.asarray(body['expression']),
                 conversion_settings=np.array(json.dumps(dict(body_iters=2,finger_iters=1,full_iters=1,
                     lie_iters=3,lod='low',batch_size=args.batch_size,source_hand_pca_components=24))))
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix('.partial.npz')
    save_soma_npz(temporary, rotations, transl, joint_names=inv.joint_names,
                  identity_model_type='smplx',identity_coeffs=np.zeros((1,10),np.float32),
                  keep_root=True,unit='meters',extra_arrays=extra)
    with np.load(temporary,allow_pickle=False) as saved:
        assert saved['poses'].shape == (frames,len(inv.joint_names),3)
        np.testing.assert_array_equal(saved['object_trans'],obj['params']['transl'])
        np.testing.assert_array_equal(saved['object_rot'],obj['params']['global_orient'])
        assert np.isfinite(saved['poses']).all()
    temporary.replace(target)
    contact = data['contact'].item()
    contact_path = args.output / 'contacts' / subject / target.name
    contact_path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(contact_path, body_contact=contact['body'], object_contact=contact['object'],
                        threshold=contact['threshold'],
                        body_topology=np.array('Original GRAB SMPL-X, NOT SOMA vertex indices'),
                        object_mesh_path=extra['package_mesh_path'])
    return dict(frames=frames, seconds=time.perf_counter()-started, bytes=target.stat().st_size,
                gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/1e6,
                mean_fit_error_mm=float(per_frame.mean()*1000),
                max_frame_mean_error_mm=float(per_frame.max()*1000), replay_checks=previews)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,default=ROOT.parent/'Grab Dataset')
    parser.add_argument('--prepared',type=Path,default=ROOT/'out/grab-preparation')
    parser.add_argument('--assets',type=Path,default=ROOT/'assets')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(exist_ok=True,parents=True)
    inventory=json.loads((args.prepared/'inventory.json').read_text())
    for record in inventory['objects']:
        folder=args.output/'objects'/record['name']; folder.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(args.prepared/'objects'/f'{record["name"]}.ply',folder/'mesh.ply')
    (args.output/'scene').mkdir(exist_ok=True)
    shutil.copyfile(args.prepared/'objects/table.ply',args.output/'scene/table.ply')
    selected=inventory['sequences']
    if args.pilot:
        selected=[s for s in selected if s['member']=='s1/mug_drink_1.npz']
    results=[]
    run_started=time.perf_counter()
    def progress(status):
        json_save(args.output/'reports/progress.json',dict(status=status,
                  converted_sequences=len(results),total_sequences=len(selected),
                  converted_frames=sum(r['frames'] for r in results),
                  total_frames=sum(r['frames'] for r in selected),
                  elapsed_seconds=time.perf_counter()-run_started))
    progress('converting')
    json_save(args.output/'manifest.json',dict(status='converting',source_dataset='GRAB',
              object_count=51,total_sequences=len(selected),fps=120.0,unit='meters',
              format='SOMA full-body compact axis-angle NPZ; includes virtual Root',
              personalized_subjects=True,subjects=inventory['subjects'],objects=inventory['objects']))
    for subject,info in inventory['subjects'].items():
        clips=[s for s in selected if s['subject']==subject]
        if not clips:continue
        print(json.dumps(dict(event='initialize_subject',subject=subject,clips=len(clips))),flush=True)
        model,soma,inv=initialize_subject(subject,info['gender'],args.prepared,args.assets,
                                         args.output,args.device,args.batch_size)
        with zipfile.ZipFile(args.input/clips[0]['archive']) as archive:
            for i,record in enumerate(clips):
                target=args.output/'motions'/subject/Path(record['member']).name
                statfile=args.output/'reports/sequences'/subject/(target.stem+'.json')
                if args.resume and target.is_file() and statfile.is_file():
                    results.append(json.loads(statfile.read_text()));continue
                if target.exists() and not args.resume:raise FileExistsError(target)
                with np.load(io.BytesIO(archive.read(record['member'])),allow_pickle=True) as data:
                    stats=convert_sequence(data,model,soma,inv,args,target,
                         record['archive']+':'+record['member'],subject)
                result=dict(record,output=target.relative_to(args.output).as_posix(),**stats)
                json_save(statfile,result); results.append(result)
                progress('converting')
                print(json.dumps(dict(event='converted',subject=subject,index=i+1,total=len(clips),
                          motion=record['member'],frames=stats['frames'],seconds=round(stats['seconds'],2),
                          mean_fit_error_mm=stats['mean_fit_error_mm'])),flush=True)
        del model,soma,inv
        import gc;gc.collect();torch.cuda.empty_cache()
    json_save(args.output/'reports/conversion.json',dict(pilot=args.pilot,sequences=len(results),
              frames=sum(r['frames'] for r in results),results=results))
    progress('conversion_finished_pending_final_validation')
    print(json.dumps(dict(event='finished',pilot=args.pilot,sequences=len(results),
                         frames=sum(r['frames'] for r in results))),flush=True)


if __name__=='__main__':main()
