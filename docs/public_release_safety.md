# Keeping local model/data files out of public Git

## Current audit finding (2026-09-17)

The migrated Linux checkout is at `~/Projects/SOMA-X`, with HEAD `69568fd`
based on upstream `d29dbe5`. The previous Windows commit `f0158f5`, which
referenced downloaded SMPL-X male/female models, is absent from this checkout.
Those full models, the neutral model and the MANO parameter files are present
locally, ignored and untracked. The index check passes. This local check does
not establish whether a previous remote copy or LFS upload has been removed.

The all-history check still flags `assets/GarmentMeasurements/point.npz`.
It was added in upstream `6d758a9` and removed in upstream `dc16291`; it is
absent from the current tree and public asset allowlist. This is an upstream
asset review finding, not evidence of a newly uploaded downloaded model or a
conclusion about its license. Establish its redistribution terms before adding
an exception, or omit that history from a separate clean publication.
**The push guard remains blocked by this historical asset.**

All 28 materialized Git LFS assets and their cached objects match their committed
SHA-256 hashes and sizes. Linux was missing Git LFS, which made those unchanged
assets appear modified. Git LFS and executable hooks were restored locally;
no model payloads, dataset contents or Git history were changed.

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

If a licensed model is committed again, a deletion commit alone is insufficient:
earlier commits still reference its LFS objects. Before publishing that history,
either prepare a fresh code-only repository or remove the affected paths from
every ref that will be published. Rewriting shared history changes commit IDs
and may require a coordinated force push. Do not rewrite or force-push a shared
branch casually. The migrated checkout already excludes the previous local
full-model commit; the remaining upstream asset finding is described above.

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

The index check passes. The history check is expected to fail until the upstream
`point.npz` asset review is resolved or that history is excluded from publication.
Source code importing SMPL-X/MANO can remain in Git; model parameters are loaded
locally at runtime. Existing local conversions and their model files are intact.
