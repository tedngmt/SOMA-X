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
cd ~/Projects/SOMA-X
```

The environment lives at `/home/nmt/miniconda3/envs/soma-x`. The editable
installation uses this checkout, so Python source edits take effect immediately.

## Recreate

```bash
conda create -n soma-x python=3.10 pip -y
conda activate soma-x
cd ~/Projects/SOMA-X
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

Outputs are in `~/Projects/SOMA-X/out/wsl-demo`. From WSL, run
`explorer.exe .` in the output directory to open it in Windows Explorer.

## Migration check (2026-09-17)

- The editable Python installation points to `~/Projects/SOMA-X`, including
  imports from outside the checkout. `pip check` passes.
- PyTorch and Warp can use the RTX 3070 Ti Laptop GPU after the move.
- Body, hand, geometry, package-layout and batched-skinning checks pass. Anny
  remains optional.
- GRAB packages are siblings under `~/Projects/`. Their recorded checksums and
  relative motion/mesh links pass.
- Git LFS was missing from Linux. Ubuntu's `git-lfs` binary was installed in
  `~/miniconda3/bin/` (already on this machine's PATH), and LFS filters were
  configured for this checkout. All 28 public assets match their Git LFS hashes.
- Local executable publication hooks were restored from `.githooks/`. See
  [public release safety](docs/public_release_safety.md) for the code-only push
  policy. The historical upstream blocker has an exact-blob exception; the push
  hook rejects new LFS pointers and never invokes an LFS upload.
- This WSL checkout reuses the existing Windows Git Credential Manager for
  GitHub authentication, configured locally for this repository. Credentials
  remain in the Windows credential store; none are written into project files.

Detailed test output is in `out/migration-validation/`. Migrated bytecode caches
were cleared so future tracebacks use the new source paths. Historical conversion
logs retain their original Windows paths as provenance.

## Earlier verification (2026-09-15)

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
