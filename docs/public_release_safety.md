# Keeping local model/data files out of public Git

## Current audit finding (2026-09-17)

The migrated Linux checkout is at `~/Projects/SOMA-X`, with local changes
based on upstream `d29dbe5`. The previous Windows commit `f0158f5`, which
referenced downloaded SMPL-X male/female models, is absent from this checkout.
Those full models, the neutral model and the MANO parameter files are present
locally, ignored and untracked. The index check passes. This local check does
not establish whether a previous remote copy or LFS upload has been removed.

The earlier history blocker, `assets/GarmentMeasurements/point.npz`, was
added in upstream `6d758a9` and removed in upstream `dc16291`. It is already
part of this fork's remote history, and absent from the current tree. The
guard now recognizes only its exact historical Git blob
`5b396c34874ce08d0b12f06a4e6a536c0b44b469`. This is an inherited-history
exception, not legal clearance for a new upload. The path remains blocked in
the index, and changed payloads remain blocked in history. No general asset
allowlist or `.gitignore` exception was added for it.

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
  history of every proposed ref, then rejects newly introduced LFS pointer blobs.
  It **never invokes the Git LFS upload hook**. Code-only updates to the existing
  fork and deletion-only pushes remain possible. LFS checkout/merge hooks remain
  available for working with the existing upstream assets locally.

The hooks are stored in `.githooks/` and installed into this clone's existing
`.git/hooks/`. Hooks are not automatically installed by cloning. On another clone,
review and copy these two hooks into the active hooks directory, preserving any
existing custom hooks and setting executable permission on Unix. Do not bypass
the guard with `--no-verify` or run `git lfs push` directly. This local push hook
does not change GitHub account billing settings or hooks on Windows/other clones.

## Code-only pushes; no LFS uploads

`tools/ci/check_no_lfs_upload.py` checks the Git objects introduced between each
destination's advertised remote commit and the proposed local commit. It checks
intermediate commits too, so adding then deleting a new LFS file still blocks.
Existing pointers already reachable from that remote commit do not require an
upload. The check never contacts an LFS server.

If the destination branch is new, all reachable history is checked. A new branch
or repository containing inherited LFS pointers is conservatively blocked; use
the existing fork branch for code-only updates, or prepare a clean code-only
export. An unknown remote commit requires a fetch before the check can proceed.

Git LFS remains installed for local asset handling. Installing it does not
subscribe to paid storage. Keeping it installed preserves correct Git status for
the upstream binary assets; removing it would make those assets look modified.
Downloaded models, data and conversion outputs remain local and ignored.

This is a path-based publication check, not a comprehensive content, secret or
license audit. Review newly allowed assets and public release contents separately.

## History cleanup is a separate operation

If a licensed model is committed again, a deletion commit alone is insufficient:
earlier commits still reference its LFS objects. Before publishing that history,
either prepare a fresh code-only repository or remove the affected paths from
every ref that will be published. Rewriting shared history changes commit IDs
and may require a coordinated force push. Do not rewrite or force-push a shared
branch casually. The migrated checkout already excludes the previous local
full-model commit; the inherited upstream exception is described above.

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

The index and history checks now pass. The separate no-LFS check runs at push
time against the actual destination branch, and rejects new LFS content.
Source code importing SMPL-X/MANO can remain in Git; model parameters are loaded
locally at runtime. Existing local conversions and their model files are intact.
