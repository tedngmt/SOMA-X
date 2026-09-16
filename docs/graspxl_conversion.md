# GraspXL MANO to SOMA-X

`tools.hand.convert_graspxl_to_soma` converts the three diverse-approach
MANO archives to right-hand `SOMAHandLayer` animations. It reads split ZIPs
directly, without joining or extracting 189 GB of input. Ordinary joined ZIPs
are also supported. Archive 1's numeric filenames and archives 2/3's
`mano_*.npy` filenames are both accepted. Robot and tabletop archives are
not handled by this converter.

## Required model

Place your licensed MANO v1.2 `MANO_RIGHT.pkl` in `assets/MANO/`.
GraspXL motion files do not contain this model. The other SOMA hand and MANO
transfer assets must also be installed under `assets/`. Use the existing
WSL `soma-x` environment described in [WSL_SETUP.md](../WSL_SETUP.md).

## Commands (WSL, from this repository)

```bash
conda activate soma-x
cd /mnt/c/Linux/SOMA-X
python -m pip install smplx chumpy

# Inspect all archives and load one sample per archive; no model required.
python -m tools.hand.convert_graspxl_to_soma --inspect

# First convert one complete sequence and review its reported error.
python -m tools.hand.convert_graspxl_to_soma --max-sequences 1

# Convert the remaining sequences, skipping completed outputs.
python -m tools.hand.convert_graspxl_to_soma --resume

# Only archive 1 records wholly contained in part00 (part01 is needed for ZIP metadata).
python -m tools.hand.convert_graspxl_to_soma --archive 1 --part 0 --resume

# First measure 20 evenly spaced sequences from that part.
python -m tools.hand.convert_graspxl_to_soma --archive 1 --part 0 --sample-count 20 --output-dir out/graspxl_part00_sample
```

The default input is `../GraspXL DataSet`; output is `out/graspxl_soma_compact`.
The default `--storage compact` uses float32 axis-angle rotations (three values
per joint) and omits duplicate original MANO motion arrays. Original motion
remains in the source archives, referenced by `source_sequence`. All frames,
25 joints, object motion, identity, and fitting-error measurements are retained.
There is no float16 quantization or PCA pose approximation.

The earlier 977–986 GB estimate applied to the initial diagnostic format,
which stored nine rotation-matrix entries per joint plus duplicate MANO arrays.
That format remains available with `--storage diagnostic`; it is not required
for SOMA-X playback.

Measured compact storage for 20 evenly spaced archive 1 part00 sequences
(3,125 frames) is 1,087,539 bytes, down from 3,305,361 bytes. Projections:

| Scope | Sequences | Compact output estimate |
|---|---:|---:|
| Archive 1 part00 | 1,691,236 | 92 GB |
| All of archive 1 | 2,154,738 | 117 GB |
| All three MANO archives | 5,964,698 | 324 GB |

These extrapolate only the measured archive 1 part00 sample, exclude filesystem
overhead, and are not guaranteed final sizes. Source archives are additional
storage. Choose a drive with sufficient space; conversion stops below 1 GiB free.
Across all 3,125 sample frames, compact replay changed mesh positions by at most
0.00164 mm (mean 0.000145 mm) relative to diagnostic matrices. The tiny change
comes from projecting numerical fitted matrices onto valid rotations. The
original MANO-to-SOMA surface fit error was 1.696 mm on average.

`--part` excludes a record that crosses the
split boundary; convert without `--part` to include every sequence. Both ZIP
parts remain necessary because the central directory is at the archive end.
Use `--input-dir`, `--output-dir`, or `--data-root` to override those locations.
Output paths preserve archive/size/object/sequence, preventing collisions
between archives. Writes are atomic per sequence. `--resume` checks stored
iteration and storage settings; use a new output directory when changing these.
Use the same assets and FPS when resuming (these are not fingerprinted).
CUDA is the default; `--device cpu` is available.
`--batch-size 256` bounds fitting memory per batch; lower it if GPU memory is limited.
The previous 32-frame default unnecessarily split typical sequences into five calls.
ZIP directory metadata still
requires memory proportional to the number of archive entries (millions here).
Only load trusted GraspXL downloads: source NPY dictionaries require pickle.

## Conversion and output

The source model matches the official visualizer: right MANO, zero shape
coefficients, full axis-angle pose, `flat_hand_mean=False`. We evaluate MANO,
transfer its wrist-centered surface to SOMA hand topology, and fit the SOMA rig.
The saved translation adds back the MANO wrist position and original translation,
preserving hand/object world alignment. No display-axis rotation is applied.

Each compressed NPZ uses `soma.io.save_soma_npz`:

- `poses`: `(T, 25, 3)` absolute local axis-angle rotations, including wrist.
  Diagnostic storage uses `(T, 25, 3, 3)` matrices instead.
- `transl`: `(T, 3)` world translation in meters.
- `identity_model_type="mano"`, zero `identity_coeffs`, `hand_type="right"`.
- `absolute_pose=True`, `keep_root=True` (retains the hand wrist).
- `object_trans`, `object_rot`: original object translation and axis-angle rotation.
- `object_id`, `object_mesh_archive`, `object_mesh_member`: original mesh reference.
- `mano_pose`, `mano_rot`, `mano_trans`: original hand motion parameters,
  included only with `--storage diagnostic`.
- `fit_mean_vertex_error_m`: mean fitting error per frame, including boundary vertices.
- Source identifier, fitting settings, and reserved `object_angle` when present.

Meshes remain in `object_dataset.zip`; no duplicate object geometry is written.
Additional unrecognized source fields are not exported. Source FPS is not
specified by the dataset card, so it is omitted unless supplied with `--fps`.
Retargeting is approximate and does not enforce collision/contact constraints.
These are hand-only animations; creating full-body grasp motions requires
additional body motion/arm placement. MANO identity is retained, so replay also
requires the licensed model.

```python
import torch
from soma.hand import SOMAHandLayer
from soma.io import load_soma_npz

data = load_soma_npz("out/graspxl_soma_compact/mano_dataset_1/large/OBJECT/1.npz")
hand = SOMAHandLayer(data_root="assets", hand_type="right",
                     identity_model_type="mano", device="cpu")
hand.prepare_identity(torch.from_numpy(data["identity_coeffs"]))
with torch.no_grad():
    frame = hand.pose(torch.from_numpy(data["poses"][:1]),
                      pose2rot=data["rotation_repr"] == "rotvec",
                      absolute_pose=True,
                      global_translation=torch.from_numpy(data["transl"][:1]))
```

## Validation and sources

Run `python -m pytest tests/hand/test_convert_graspxl_to_soma.py -q`.
Archive-boundary, path, and data-contract tests run without model assets.
The numerical MANO/SOMA reconstruction test skips until `MANO_RIGHT.pkl` is installed.
Local validation with the supplied v1.2 model: all nine tests passed, covering
compact and diagnostic reconstruction, split ZIP boundaries, and data validation; the first
real 155-frame sequence converted on CUDA with 1.668 mm mean surface fitting error.

- [GraspXL dataset card](https://huggingface.co/datasets/ethHuiZhang/GraspXL)
- [Official MANO visualizer](https://github.com/zdchan/GraspXL_visualization/blob/main/scripts/visualizer_mano.py)

GraspXL is CC BY-NC 4.0. Retain its attribution and cite Zhang et al.,
*GraspXL: Generating Grasping Motions for Diverse Objects at Scale*, ECCV 2024.
MANO's separate model license also applies.

## Local CUDA and throughput measurements (2026-09-16)

The WSL `soma-x` environment already uses the RTX 3070 Ti Laptop GPU (8 GB),
PyTorch `2.10.0+cu128` with CUDA runtime 12.8, and Warp 1.11.1 (built with CUDA
12.9). The Windows driver is 581.57. Windows `nvcc --version` reports toolkit
11.8, but that is a separate compiler installation, not the runtime used by this
conversion. No toolkit upgrade is required to enable GPU conversion here.

With identical fitting iterations and float32 precision, eight representative
sequences took 0.691 seconds/sequence at frame batch 32 and 0.222 seconds at
batch 256 (3.1x faster for the conversion call). Maximum per-frame fitting-error
change was 0.000062 mm. Batch 256 is now the default; it uses about 1.05 GB of
PyTorch-allocated memory on these samples, excluding Warp/driver allocations.

A separate test read 20 sequences from the original split ZIP and included
NPY loading, fitting, compact compression, output writes, and progress logging.
Two passes took 0.252–0.264 seconds/sequence after the one-time ZIP index load.
That projects to approximately 5 days for archive 1 part00, 6.3–6.6 days for
archive 1, and 17.4–18.2 days for all archives. These short sample projections
assume continuous operation; laptop thermals, power mode, other workloads,
filesystem growth, and motion variation can change sustained throughput.
This supersedes the earlier 47-day projection using 32-frame batches.

A separate cross-sequence batching prototype processed the same 20 saved
diagnostic samples in 0.197 seconds/sequence at batch 512 (about 13.6 days if
extrapolated). That test includes reading NPZ samples and writing compact output,
but not the original ZIP input path, so its projection is not the current
converter's end-to-end estimate. Replay differed from the batch-32 baseline by
at most 0.00087 mm on the eight shared sequences. This prototype is not yet
integrated into the dataset converter; simply setting `--batch-size 512` does
not combine separate sequences. The 1024-frame experiment exhausted almost all
8 GB of device memory and is not recommended for this laptop.

Reports in this workspace: `out/graspxl-benchmark/report.json` (batch sweep)
and `out/graspxl-benchmark/e2e/report.json` (original ZIP end-to-end test).

## Exact local motion inventory (2026-09-16)

| Archive (both parts) | Sequences | Distinct object IDs |
|---|---:|---:|
| `mano_dataset_1` | 2,154,738 | 261,323 |
| `mano_dataset_2` | 1,905,236 | 260,847 |
| `mano_dataset_3` | 1,904,724 | 260,804 |
| Combined | 5,964,698 | 262,812 |

An object ID can appear in multiple sizes and archives. The combined unique
count is a union, not the sum of the rows. Counting `(size, object_id)` pairs
gives 442,843 distinct scaled variants across all archives. These counts describe
objects referenced by motion records, not a check of object mesh availability.
The full per-part and per-size inventory is in `out/graspxl-inventory/README.md`
and `out/graspxl-inventory/inventory.json`. Only the six MANO split files were
present in the input folder at the time of this inventory.
