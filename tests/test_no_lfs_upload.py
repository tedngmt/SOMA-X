"""Exercise push checks against real Git history without installing Git LFS."""

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

TOOLS_CI = Path(__file__).resolve().parents[1] / "tools" / "ci"
ZERO_OID = "0" * 40


def _pointer(digest: str = "a") -> bytes:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{digest * 64}\n"
        "size 12345\n"
    ).encode("ascii")


class GitRepo:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir()
        self.git("init", "--initial-branch=main")

    def git(self, *args: str) -> str:
        result = subprocess.run(
            [
                "git",
                "-c", "core.hooksPath=" + os.devnull,
                "-c", "user.name=Guard Test",
                "-c", "user.email=guard-test@example.invalid",
                *args,
            ],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def commit(self, files: dict[str, bytes | None]) -> str:
        for name, content in files.items():
            path = self.root / name
            if content is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        self.git("add", "--all")
        self.git("commit", "--quiet", "-m", "Test fixture")
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Do not inherit the user's LFS filters, identity, hooks, or repository paths.
    for name in tuple(os.environ):
        if name.startswith("GIT_CONFIG_") or name in {
            "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_ATTR_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_LFS_SKIP_SMUDGE", "1")
    monkeypatch.syspath_prepend(str(TOOLS_CI))
    guard = import_module("check_no_lfs_upload")
    repo = GitRepo(tmp_path / "repo")
    monkeypatch.setattr(guard, "ROOT", repo.root)
    return repo, guard


def _update(local: str, remote: str) -> list[str]:
    return [f"refs/heads/main {local} refs/heads/main {remote}\n"]


def test_code_only_child_of_existing_remote_pointer_is_allowed(repository):
    repo, guard = repository
    remote = repo.commit({"assets/existing.npz": _pointer()})
    local = repo.commit({"tools/example.py": b"print('source change')\n"})

    assert guard.check_updates(iter(_update(local, remote))) is None


@pytest.mark.parametrize("already_tracked", [False, True])
def test_new_or_modified_pointer_is_blocked(repository, already_tracked):
    repo, guard = repository
    baseline = {"README.md": b"Source repository\n"}
    if already_tracked:
        baseline["assets/model.npz"] = _pointer("a")
    remote = repo.commit(baseline)
    local = repo.commit({"assets/model.npz": _pointer("b")})

    with pytest.raises(ValueError):
        guard.check_updates(_update(local, remote))


def test_pointer_added_then_deleted_in_outgoing_history_is_blocked(repository):
    repo, guard = repository
    remote = repo.commit({"README.md": b"Source repository\n"})
    repo.commit({"assets/model.npz": _pointer()})
    local = repo.commit({"assets/model.npz": None})

    with pytest.raises(ValueError):
        guard.check_updates(_update(local, remote))


def test_new_remote_branch_checks_entire_pointer_history(repository):
    repo, guard = repository
    repo.commit({"assets/model.npz": _pointer()})
    local = repo.commit({"tools/example.py": b"print('source change')\n"})

    with pytest.raises(ValueError):
        guard.check_updates(_update(local, ZERO_OID))


def test_deletion_only_push_is_allowed(repository):
    repo, guard = repository
    remote = repo.commit({"assets/model.npz": _pointer()})

    assert guard.check_updates(
        [f"(delete) {ZERO_OID} refs/heads/main {remote}\n"]
    ) is None


def test_unknown_nonzero_remote_commit_fails_closed(repository):
    repo, guard = repository
    local = repo.commit({"README.md": b"Source repository\n"})

    with pytest.raises(ValueError):
        guard.check_updates(_update(local, "e" * 40))


def test_source_containing_pointer_header_literal_is_allowed(repository):
    repo, guard = repository
    remote = repo.commit({"README.md": b"Source repository\n"})
    source = b'POINTER_EXAMPLE = """\n' + _pointer() + b'"""\n'
    local = repo.commit({"tools/pointer_example.py": source})

    assert guard.check_updates(_update(local, remote)) is None


def test_new_code_only_repository_is_allowed(repository):
    repo, guard = repository
    local = repo.commit({"README.md": b"Source repository\n"})

    assert guard.check_updates(_update(local, ZERO_OID)) is None


def test_historical_exception_allows_only_exact_upstream_blob(repository, monkeypatch):
    repo, _ = repository
    guard = import_module("check_public_release")
    monkeypatch.setattr(guard, "ROOT", repo.root)
    monkeypatch.setattr(sys, "argv", ["guard", "--history"])
    path = "assets/GarmentMeasurements/point.npz"
    upstream_pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:6ae75a1ab7a8ae4f46bac503146976ba59cf918bca8aefe7320a3ed02aa2416a\n"
        b"size 8626920\n"
    )
    repo.commit({path: upstream_pointer})
    repo.commit({path: None, "README.md": b"Upstream removed the old asset.\n"})
    assert guard.main() == 0

    # Reusing the historical filename for different content must still block.
    repo.commit({path: _pointer("b")})
    assert guard.main() == 1


def test_historical_exception_does_not_allow_index_additions(repository, monkeypatch):
    repo, _ = repository
    guard = import_module("check_public_release")
    monkeypatch.setattr(guard, "ROOT", repo.root)
    monkeypatch.setattr(sys, "argv", ["guard", "--index"])
    repo.commit({"assets/GarmentMeasurements/point.npz": _pointer()})

    assert guard.main() == 1
