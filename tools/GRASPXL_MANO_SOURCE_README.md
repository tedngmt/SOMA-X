# Selected GraspXL MANO sources

This package contains the **749 original MANO motions (116,095 frames)** used to create the paired `GraspXL_SOMA_51Objects_GRABMatched` package. It covers the same **51 object-and-size entries, using 48 unique object IDs**, with the matching meshes. The original `.npy` files are extracted unchanged from the three diverse-approach MANO archives; they are not reconstructed from SOMA.

## Files and pairing

- `motions/<object_key>/mano_dataset_N/<sequence>.npy`: original hand and object motion.
- `objects/<object_key>/mesh.obj`: the selected size variant's original mesh.
- `manifest.json`: object selection, mesh paths, and motion paths.
- `metadata/pairs.json`: one record per motion, giving the original MANO path, matching SOMA path, frame count, and archive provenance. Paths are relative to their respective package roots.
- `scripts/prepare_viewer.py`: inspect a pair or prepare one motion for the official MANO viewer.

Keep the MANO and SOMA package folders beside each other. Object numbering and directory keys match, and frame `i` in the MANO file corresponds to frame `i` in its SOMA file. The selection prioritizes GRAB's object types; its labelled substitutes and size differences still apply. See the SOMA package's `COMPARISON.md` and `MOTION_COUNTS.md` for the selection details.

## Inspect a matching pair

Requires Python 3.9+ and NumPy. From this package's directory, inspect the first handled-mug motion:

```text
python scripts/prepare_viewer.py --object-number 29 --sequence-index 0
```

The command prints the paired MANO/SOMA paths and shapes and checks that frame counts, object IDs, and object trajectories agree. Sequence indices start at zero and follow `manifest.json`'s `motion_paths` order. Use `--package PATH` when running from another location. These checks do not replace visual comparison of the hand surfaces.

## View the original MANO motion

Set up the [official GraspXL viewer](https://github.com/zdchan/GraspXL_visualization) and its dependencies according to its instructions. Then stage one selected motion and its mesh:

```text
python scripts/prepare_viewer.py --object-number 29 --sequence-index 0 --viewer-root "C:/path/to/GraspXL_visualization"
```

The helper copies these two files unchanged into the viewer's `data/GraspXL/recorded/` and `data/GraspXL/object_mesh/` directories, using a unique name containing the object key, archive, and sequence. Identical existing files are kept; different existing files are never overwritten. It prints the exact viewer command to run **from the viewer repository directory**. It does not install or launch the viewer.

The [official MANO viewer](https://github.com/zdchan/GraspXL_visualization/blob/main/scripts/visualizer_mano.py) uses the right MANO hand with `use_pca=False`, `flat_hand_mean=False`, and zero shape coefficients. It reconstructs the hand from `rot` and `pose`, then adds `trans`. Use these same settings when comparing against SOMA. MANO and SOMA pose arrays describe different skeletons, so compare reconstructed geometry at matching frame indices rather than subtracting their pose coefficients. Keep mesh scale and coordinate transforms consistent; the official viewer also centers and rotates both meshes for display.

Your licensed MANO model is **not bundled**. Configure it in the viewer's documented model directory (`data/body_models/mano`). The `.npy` dictionaries require `allow_pickle=True`; use that only for trusted source files such as these originals.

## Timing and scope

The downloaded motions do not identify their frame rate. The upstream [generator](https://github.com/zdchan/GraspXL/blob/main/raisimGymTorch/raisimGymTorch/env/envs/ours_demo/objaverse_demo.py) and [default configuration](https://github.com/zdchan/GraspXL/blob/main/raisimGymTorch/raisimGymTorch/env/envs/ours_demo/cfgs/cfg_reg.yaml) suggest 100 FPS, but that does not establish the configuration used for these files. FPS therefore remains unknown. Compare by frame index first, and use the same playback speed for both representations.

This is the diverse-approach **hand-and-object** subset. It does not contain full-body motion. The 112 additional raw tabletop motions are separate extras in the SOMA package's `extra_assets/` directory; they are not part of these 749 pairs and use a different MANO convention. Numerical agreement does not establish physically valid contact or readiness for Isaac training.

Source: [GraspXL dataset](https://huggingface.co/datasets/ethHuiZhang/GraspXL) and [project repository](https://github.com/zdchan/GraspXL). Keep the source dataset attribution and license conditions, as well as the separate MANO model license, when using or sharing the data.
