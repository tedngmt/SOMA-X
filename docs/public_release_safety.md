# Keeping local model/data files out of public Git

## Current audit finding (2026-09-16)

Commit `f0158f5` added Git LFS pointers for these separately downloaded full models:

- `assets/SMPLX/SMPLX_MALE.npz` (108,753,445 bytes)
- `assets/SMPLX/SMPLX_FEMALE.npz` (108,794,146 bytes)

It also added `assets/SMPLX/version.txt`. Both `main` and the local cached
`origin/main` reference contained that commit during this audit. Remote server
visibility and whether the LFS payloads were uploaded have not been checked.

The all-history audit also flags an older `assets/GarmentMeasurements/point.npz`
payload. This asset is absent from the current tree and not in the current public
asset allowlist. That is a review finding, not a conclusion about its license;
exclude it from a clean publication or establish its redistribution terms first.

The three `assets/SMPLX/` paths have been removed from the current Git index while the actual
local files are retained. **That does not erase the earlier commit or remotely
stored LFS objects. The current history is still blocked from publication.**

## Protection added

- `.gitignore` excludes downloaded model parameters, motion arrays, checkpoints,
  archives, exported meshes and local data/output directories.
- Exact exceptions preserve assets already supplied by upstream SOMA-X before
  these downloads. Their own licenses and existing attribution still apply;
  an exception does not authorize replacing that file with a licensed model.
- `tools/ci/check_public_release.py --index` checks the whole proposed commit
  tree, including files forcibly added despite `.gitignore`.
- `--history` checks HEAD's reachable history; `--all-history` also checks other
  local branches, tags and remote-tracking refs.
- The local pre-commit hook checks the index. The pre-push hook checks the full
  history of every proposed ref **before** invoking the existing Git LFS upload.
  Deletion-only pushes remain possible. Existing LFS checkout/merge hooks remain.

The hooks are stored in `.githooks/` and installed into this clone's existing
`.git/hooks/`. Hooks are not automatically installed by cloning. On another clone,
review and copy these two hooks into the active hooks directory, preserving any
existing custom hooks and setting executable permission on Unix. Do not bypass
the guard with `--no-verify`, or use `git lfs push` directly before history cleanup.

This is a path-based publication check, not a comprehensive content, secret or
license audit. Review newly allowed assets and public release contents separately.

## History cleanup is a separate operation

A new deletion commit alone is insufficient: earlier commits still reference the
full-model LFS objects. Before publishing, either prepare a fresh code-only
repository with clean history, or remove the affected paths from every ref that
will be published. Rewriting shared history changes commit IDs and may require a
coordinated force push. Do not rewrite or force-push a shared branch casually.

If models reached a hosting service, also address its retained LFS objects,
cached views and other copies. Rewriting local Git history is not proof that the
host has deleted the uploaded content. See the hosting service's removal process.

## Why full models are excluded

The official [SMPL-X model license](https://smpl-x.is.tue.mpg.de/modellicense.html)
and [MANO license](https://mano.is.tue.mpg.de/license.html) contain restrictions on
distribution. The full model parameter downloads are distinct from public
correspondence meshes or exports under a separate body license. Keep the
repository's `LICENSE`, `ATTRIBUTIONS.MD`, and upstream attribution intact.

Raw GRAB/GraspXL, derived packages and personalized templates remain local. The
current sibling dataset folders are outside this repository; copies made inside
the repository are covered by the ignore rules and guard where their paths or
file types match. Sharing derived data requires its own license review.

## Before committing or publishing

```text
python tools/ci/check_public_release.py --index
python tools/ci/check_public_release.py --history
git diff --cached --stat
```

The index check should pass after untracking the downloaded models. The history
check is expected to fail until the historical model references are removed.
Source code importing SMPL-X/MANO can remain in Git; model parameters are loaded
locally at runtime. Existing local conversions and their model files are intact.
