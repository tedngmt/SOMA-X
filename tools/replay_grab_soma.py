"""Replay a converted GRAB sequence with its personalized SOMA identity."""
import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

from soma.body import SOMALayer


def load_personalized_layer(package, motion, assets, device='cpu'):
    template = trimesh.load(package / str(motion['subject_template_path'].item()),
                            process=False, maintain_order=True)
    layer = SOMALayer(assets, lod='low', identity_model_type='smplx',
                      identity_model_kwargs={'gender': str(motion['gender'].item())},
                      device=device, mode='dense' if device == 'cpu' else 'warp')
    layer.identity_model.identity_model.v_template = torch.as_tensor(
        np.asarray(template.vertices), dtype=torch.float32, device=device)
    layer.prepare_identity(torch.zeros(1, 10, device=device))
    return layer


def replay(layer, motion, indices):
    device = layer.identity_model.identity_model.v_template.device
    poses = torch.as_tensor(motion['poses'][indices, 1:], device=device)
    transl = torch.as_tensor(motion['transl'][indices], device=device)
    with torch.no_grad():
        return layer.pose(poses, transl=transl, pose2rot=True,
                          absolute_pose=True, apply_correctives=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--motion', required=True, help='Relative motion NPZ path')
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.package / args.motion, allow_pickle=False) as source:
        motion = dict(source)
    indices = np.array([0, len(motion['poses']) // 2, len(motion['poses']) - 1])
    layer = load_personalized_layer(args.package, motion, args.assets, args.device)
    result = replay(layer, motion, indices)
    arrays = {k: result[k].cpu().numpy() for k in ('vertices', 'joints', 'transforms')}
    assert all(np.isfinite(v).all() for v in arrays.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, frame_indices=indices,
                        faces=layer.faces.cpu().numpy())
    print(f'Replayed {indices.tolist()}; vertices {arrays["vertices"].shape}; {args.output}')


if __name__ == '__main__':
    main()
