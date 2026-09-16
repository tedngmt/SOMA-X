# Local Ubuntu / WSL setup

This checkout is installed in the `soma-x` Conda environment in Ubuntu.

## Activate

From Windows PowerShell, open Ubuntu:

```powershell
wsl -d Ubuntu
```

Then in Ubuntu:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate soma-x
cd /mnt/c/Linux/SOMA-X
```

The environment lives at `/home/nmt/miniconda3/envs/soma-x`. The editable
installation uses this checkout, so Python source edits take effect immediately.

## Recreate

```bash
conda create -n soma-x python=3.10 pip -y
conda activate soma-x
cd /mnt/c/Linux/SOMA-X
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev,demo]' numpy==1.23.5 scipy==1.15.3 warp-lang==1.11.1 trimesh==4.11.1 rtree==1.4.1 cholespy==2.2.0 usd-core==26.3
```

The model assets are already present in `assets/`. Optional SMPL and MANO
backends need separately licensed model files; Anny needs its optional extra.

## Check the installation

```bash
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
python -m pytest tests/test_soma_layer.py tests/hand/test_soma_hand_layer.py -m 'not slow' -q
```

## Render a short body demo

```bash
python tools/demo_soma_vis.py --identity-model-type soma,mhr --device cuda:0 --max-frames 16 --image-size 1920 --pose-batch-size 8 --output-dir out/wsl-demo
```

Outputs are accessible from Windows at `C:\Linux\SOMA-X\out\wsl-demo`.

## Verified on this computer

- Python 3.10.21; PyTorch 2.10.0+cu128; Warp 1.11.1.
- PyTorch and Warp detected the NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB).
- `pip check`: no broken requirements.
- Selected body, hand, geometry, package-layout, and batched-skinning tests:
  **96 passed, 1 skipped, 49 deselected**. The skip is the optional Anny backend;
  tests marked `slow` were excluded.
- Rendered 16-frame SOMA and MHR videos successfully in `out/wsl-demo/`.
- WSL's EGL renderer reported a graphics-device permission warning and used
  its fallback renderer successfully. CUDA model computation works.

Run `bash out/verify_wsl.sh` from the repo to repeat these checks. Test logs,
JUnit results, and the installed package list are in `out/wsl-validation/`.
