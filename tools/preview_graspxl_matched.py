"""Render saved SOMA hand and paired object on three frames for visual QA."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-arrays', type=Path)
    args = parser.parse_args()
    path = next((args.package / 'motions').glob('*_mug_*/mano_dataset_1/1.npz'))
    with np.load(path, allow_pickle=False) as z:
        data = dict(z)
    indices = [0, len(data['poses'])//2, len(data['poses'])-1]
    if args.render_arrays is None:
        import torch
        from soma.hand import SOMAHandLayer
        hand = SOMAHandLayer(data_root=str(ROOT / 'assets'), hand_type='right',
                             identity_model_type='mano', device='cpu')
        with torch.no_grad():
            output = hand(poses=torch.from_numpy(data['poses'][indices]),
                          identity_coeffs=torch.from_numpy(data['identity_coeffs']).expand(3,-1),
                          pose2rot=True, absolute_pose=True,
                          global_translation=torch.from_numpy(data['transl'][indices]))
        args.output.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(args.output, vertices=output['vertices'].cpu().numpy(), faces=hand.faces.cpu().numpy())
        print(args.output)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sys.path.insert(0, str(ROOT / 'out/household-pilot'))
    from render_candidates import parse_obj
    with np.load(args.render_arrays) as replay:
        vertices = replay['vertices']; faces = replay['faces']
    object_v, object_f = parse_obj((args.package / str(data['package_mesh_path'])).read_bytes())
    fig = plt.figure(figsize=(15,9), layout='constrained')
    def colors(v,f,color):
        tri=v[f];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-12)
        light=np.array([.4,-.5,.75]);light/=np.linalg.norm(light)
        intensity=.35+.65*np.abs(norm@light)
        return intensity[:,None]*np.array(color)[None,:]
    for panel,index in enumerate(indices+indices):
        obj = object_v @ Rotation.from_rotvec(data['object_rot'][index]).as_matrix().T + data['object_trans'][index]
        body = vertices[panel%3]
        ax = fig.add_subplot(2,3,panel+1,projection='3d')
        ax.add_collection3d(Poly3DCollection(obj[object_f], facecolors=colors(obj,object_f,[.4,.65,.81]),edgecolors='none',alpha=.8))
        ax.add_collection3d(Poly3DCollection(body[faces], facecolors=colors(body,faces,[.9,.67,.47]),edgecolors='none'))
        both=np.concatenate([body,obj]);lo=both.min(axis=0);hi=both.max(axis=0)
        center=(lo+hi)/2; radius=max(hi-lo)*.58
        ax.set_xlim(center[0]-radius,center[0]+radius)
        ax.set_ylim(center[1]-radius,center[1]+radius)
        ax.set_zlim(center[2]-radius,center[2]+radius)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-65 if panel<3 else 115);ax.set_axis_off()
        ax.set_title(f'Frame {index} / {len(data["poses"])-1} | view {1 if panel<3 else 2}')
    fig.suptitle('Converted SOMA right hand with original mug trajectory | source FPS unspecified')
    args.output.parent.mkdir(exist_ok=True,parents=True)
    fig.savefig(args.output,dpi=120,facecolor='white')
    print(args.output)


if __name__=='__main__':main()
