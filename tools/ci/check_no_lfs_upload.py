"""Allow code pushes while rejecting any newly introduced Git LFS pointers.

Reads Git's pre-push ref updates. Uses only local Git objects and the advertised
remote commit; it never contacts an LFS server or uploads files. Existing remote
LFS history can remain, but a new destination containing LFS history is blocked.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OID = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")
POINTER_HEADERS = (
    b"version https://git-lfs.github.com/spec/v1\n",
    b"version https://hawser.github.com/spec/v1\n",
)


def git(*args, input=None):
    result = subprocess.run(
        ["git", *args], cwd=ROOT, input=input, capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def check_updates(lines):
    if git("rev-parse", "--is-shallow-repository").strip() == b"true":
        raise ValueError("Fetch complete history before checking for new LFS files.")
    candidates = set()
    for line in lines:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("Invalid pre-push ref update.")
        _, local_oid, _, remote_oid = fields
        if not OID.fullmatch(local_oid) or not OID.fullmatch(remote_oid):
            raise ValueError("Invalid pre-push object ID.")
        if set(local_oid) == {"0"}:
            continue
        revisions = [local_oid]
        if set(remote_oid) != {"0"}:
            try:
                git("cat-file", "-e", remote_oid)
            except ValueError as error:
                raise ValueError(
                    "Remote commit is unavailable locally. Fetch the destination "
                    "branch and retry; no LFS upload was attempted."
                ) from error
            revisions.append("^" + remote_oid)
        for record in git("rev-list", "--objects", *revisions).splitlines():
            candidates.add(record.split(b" ", 1)[0])
    if not candidates:
        return
    sizes = git(
        "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input=b"\n".join(sorted(candidates)) + b"\n",
    )
    blocked = []
    for line in sizes.splitlines():
        oid, kind, size = line.split()
        # Git LFS pointers are small text blobs. Large payloads remain subject
        # to the separate publication guard; never load model payloads here.
        if kind != b"blob" or int(size) > 1024:
            continue
        content = git("cat-file", "blob", oid.decode()).replace(b"\r\n", b"\n")
        if content.startswith(POINTER_HEADERS):
            blocked.append(oid.decode())
    if blocked:
        raise ValueError(
            f"{len(blocked)} new Git LFS pointer(s) would be published. "
            "This checkout permits code-only pushes and does not upload LFS files. "
            "Remove new asset commits from the proposed history, or use a clean "
            "code-only export for a new remote. Pointer blobs: " + ", ".join(blocked)
        )


def main():
    try:
        check_updates(sys.stdin)
    except ValueError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 1
    print("PASS: no new Git LFS pointers; no LFS upload will run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
