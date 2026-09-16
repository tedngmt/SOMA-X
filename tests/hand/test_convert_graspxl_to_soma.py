"""Archive integrity and data-contract checks for GraspXL conversion."""

import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from tools.hand.convert_graspxl_to_soma import SplitReader, archives, sequence_path, validate_sequence
from tools.hand.convert_graspxl_to_soma import selected_members


def test_part_selection_excludes_crossing_record():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i in range(3):
            archive.writestr(f"large/o/{i}.npy", b"sample" * 20)
    with zipfile.ZipFile(buffer) as archive:
        entries = archive.infolist()
        boundary = entries[1].header_offset + 40
        assert [x.filename for x in selected_members(archive, 0, boundary)] == [entries[0].filename]
        assert [x.filename for x in selected_members(archive, boundary)] == [entries[2].filename]
        assert list(selected_members(archive, 0, entries[1].header_offset)) == entries[:1]


def test_split_zip_crosses_part_boundaries(tmp_path):
    buffer = io.BytesIO()
    payload = bytes(range(256)) * 20
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("small/object/mano_0.npy", payload)
    raw = buffer.getvalue()
    parts = []
    for index, content in enumerate((raw[:17], raw[17:211], raw[211:])):
        path = tmp_path / f"part{index}"
        path.write_bytes(content)
        parts.append(path)
    with SplitReader(parts) as stream:
        stream.seek(10)
        assert stream.read(300) == raw[10:310]
        stream.seek(-7, 2)
        assert stream.read() == raw[-7:]
        stream.seek(len(raw) + 20)
        assert stream.read() == b""
        with zipfile.ZipFile(stream) as archive:
            assert archive.read("small/object/mano_0.npy") == payload
    assert all(file.closed for file in stream.files)


@pytest.mark.parametrize("name", ["../small/o/mano_0.npy", "/small/o/mano_0.npy", "C:/small/o/mano_0.npy"])
def test_unsafe_members(name):
    with pytest.raises(ValueError):
        sequence_path(name)


def test_layout_and_missing_parts(tmp_path):
    assert sequence_path("mano_dataset_1/large/o/1.npy") == Path("large/o/1.npy")
    assert sequence_path("outer/small/o/mano_0.npy") == Path("small/o/mano_0.npy")
    assert sequence_path("small/o/allegro_0.npy") is None
    (tmp_path / "mano_dataset_1.zip.part01").touch()
    with pytest.raises(ValueError, match="Missing"):
        archives(tmp_path)


def test_sequence_contract():
    data = {"right_hand": {"trans": np.zeros((2, 3)), "rot": np.zeros((2, 3)),
                           "pose": np.zeros((2, 45))},
            "o": {"trans": np.zeros((2, 3)), "rot": np.zeros((2, 3))}}
    assert validate_sequence(data) == ("o", 2)
    data["right_hand"]["pose"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        validate_sequence(data)
    data["right_hand"]["pose"] = np.zeros((3, 45))
    with pytest.raises(ValueError, match="frame"):
        validate_sequence(data)


@pytest.mark.parametrize("storage", ["compact", "diagnostic"])
def test_real_conversion_and_world_alignment(tmp_path, storage):
    """Exercise actual MANO -> SOMA -> NPZ reconstruction when licensed assets exist."""
    from argparse import Namespace
    from tools.hand.convert_graspxl_to_soma import Converter

    assets = Path(__file__).resolve().parents[2] / "assets"
    if not (assets / "MANO/MANO_RIGHT.pkl").exists():
        pytest.skip("Licensed MANO_RIGHT.pkl is not installed")
    import torch
    from soma.io import load_soma_npz

    args = Namespace(data_root=assets, device="cpu", batch_size=2, bcd_iters=1, lie_iters=3, fps=None,
                     storage=storage)
    converter = Converter(args)
    data = {"right_hand": {"trans": np.array([[1., 2., 3.], [1.1, 2., 3.]]),
                           "rot": np.array([[0., 0., 0.], [0.2, -0.1, 0.3]]),
                           "pose": np.zeros((2, 45))},
            "o": {"trans": np.zeros((2, 3)), "rot": np.zeros((2, 3))}}
    path = tmp_path / "hand.npz"
    converter.convert(data, path, "test", Path("small/o/mano_0.npy"))
    saved = load_soma_npz(path)
    assert saved["poses"].shape == ((2, 25, 3) if storage == "compact" else (2, 25, 3, 3))
    assert ("mano_pose" in saved) == (storage == "diagnostic")
    assert saved["absolute_pose"] and saved["keep_root"]
    np.testing.assert_array_equal(saved["object_trans"], data["o"]["trans"])
    with torch.no_grad():
        converter.hand.prepare_identity(converter.betas.expand(2, -1))
        recon = converter.hand.pose(torch.from_numpy(saved["poses"]), pose2rot=storage == "compact",
                                    absolute_pose=True, global_translation=torch.from_numpy(saved["transl"]))
        source = converter.mano(global_orient=torch.tensor(data["right_hand"]["rot"]).float(),
                                hand_pose=torch.zeros(2, 45), betas=torch.zeros(2, 10))
        target = converter.hand.identity_model._to_soma_interp(source.vertices - source.joints[:, :1])
        target += source.joints[:, :1] + torch.tensor(data["right_hand"]["trans"]).float()[:, None]
        error = (target - recon["vertices"]).norm(dim=-1).mean(dim=-1).numpy()
    np.testing.assert_allclose(error, saved["fit_mean_vertex_error_m"], atol=1e-5)
    assert np.max(error) < 0.02
