"""Tests of the cryoDRGN command wrappers and star helpers that the jobs do not cover."""

from __future__ import annotations

import pickle
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
from fake_cryodrgn import FakeCryoDRGN
from starfile_rs import read_star

from himena_relion_cryodrgn import _impl

if TYPE_CHECKING:
    from conftest import RelionProject


def _write_tilt_star(path, groups: list[tuple[str, int, int]]) -> None:
    """Write a parse_relion-like star of (rlnGroupName, n_tilts, rlnOriginalParticle)."""
    lines = [
        "data_\n",
        "loop_",
        "_rlnImageName #1",
        "_rlnGroupName #2",
        "_rlnOriginalParticle #3",
    ]
    for name, n_tilts, original in groups:
        lines += [f"{i + 1:06d}@{name}.mrcs {name} {original}" for i in range(n_tilts)]
    path.write_text("\n".join(lines) + "\n")


def test_read_particle_image_params(relion_project: RelionProject):
    params = _impl.read_particle_image_params(relion_project.particles_star)

    assert params.box_size == relion_project.image_size
    assert params.pixel_size == pytest.approx(relion_project.image_pixel_size)


def test_backproject_without_tilt(tmp_path, fake_cryodrgn: FakeCryoDRGN):
    for name in ["particles.star", "ctf.pkl", "poses.pkl"]:
        (tmp_path / name).touch()

    result = _impl.cryodrgn_backproject_voxel(
        particles_star=tmp_path / "particles.star",
        ctf_file=tmp_path / "ctf.pkl",
        poses_file=tmp_path / "poses.pkl",
        output_dir=tmp_path / "backproject",
        dose_per_tilt=None,
    )

    assert result.indices_file is None
    bp = fake_cryodrgn.parsed("backproject_voxel")
    assert not bp.tilt
    assert not bp.force_ntilts


@pytest.mark.parametrize("ntilts, expected", [(10, [0]), (5, [0, 1])])
def test_backproject_indices_file(
    tmp_path, fake_cryodrgn: FakeCryoDRGN, ntilts: int, expected: list[int]
):
    _write_tilt_star(tmp_path / "particles.star", [("p1", 12, 1), ("p2", 6, 2)])
    for name in ["ctf.pkl", "poses.pkl"]:
        (tmp_path / name).touch()

    result = _impl.cryodrgn_backproject_voxel(
        particles_star=tmp_path / "particles.star",
        ctf_file=tmp_path / "ctf.pkl",
        poses_file=tmp_path / "poses.pkl",
        output_dir=tmp_path / "backproject",
        dose_per_tilt=3.0,
        ntilts=ntilts,
    )

    assert result.indices_file is not None
    assert (tmp_path / "backproject" / f"indices_force_ntilts_{ntilts}.pkl").samefile(
        result.indices_file
    )
    assert _impl.load_indices(result.indices_file).tolist() == expected


def test_save_and_load_indices(tmp_path):
    _impl.save_indices(tmp_path / "ind.pkl", np.array([3, 1, 2]))
    loaded = _impl.load_indices(tmp_path / "ind.pkl")
    assert loaded.dtype.kind == "i"
    assert loaded.tolist() == [3, 1, 2]

    # Plain Python ints, loadable by cryoDRGN with any numpy version.
    with open(tmp_path / "ind.pkl", "rb") as f:
        raw = pickle.load(f)
    assert type(raw) is list
    assert all(type(i) is int for i in raw)


def test_particle_indices_to_star_rows(tmp_path):
    # Row 2 of particles.star (rlnOriginalParticle 2) was skipped by parse_relion.
    star = tmp_path / "particles_2d.star"
    _write_tilt_star(star, [("TS_01/1", 3, 1), ("TS_01/3", 2, 3), ("TS_01/4", 3, 4)])

    assert _impl.particle_indices_to_star_rows(star, [0, 1, 2]).tolist() == [0, 2, 3]
    # sorted and unique, to keep the order of particles.star
    assert _impl.particle_indices_to_star_rows(star, [2, 0, 2]).tolist() == [0, 3]
    assert _impl.particle_indices_to_star_rows(star, []).tolist() == []
    with pytest.raises(IndexError):
        _impl.particle_indices_to_star_rows(star, [3])


def test_particle_indices_to_star_rows_requires_original_particle(tmp_path):
    star = tmp_path / "particles_2d.star"
    star.write_text("data_\n\nloop_\n_rlnImageName #1\n_rlnGroupName #2\na.mrcs p1\n")

    with pytest.raises(ValueError, match="rlnOriginalParticle"):
        _impl.particle_indices_to_star_rows(star, [0])


def test_restore_star_blocks(tmp_path):
    original = tmp_path / "original.star"
    original.write_text(
        "data_general\n\n_rlnTomoSubTomosAre2DStacks 1\n\n"
        "data_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnImageSize #2\n1 64\n\n"
        "data_particles\n\nloop_\n_rlnTomoParticleName #1\n_rlnOpticsGroup #2\n"
        "TS_01/1 1\nTS_01/2 1\n"
    )
    filtered = tmp_path / "filtered.star"  # as written by cryodrgn_utils filter_star
    filtered.write_text(
        "# Created by cryoDRGN\n\n"
        "data_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnImageSize #2\n1 64\n\n\n"
        "data_particles\n\nloop_\n_rlnTomoParticleName #1\n_rlnOpticsGroup #2\n"
        "TS_01/2 1\n"
    )

    _impl.restore_star_blocks(original, filtered)

    star = read_star(filtered)
    assert list(star.keys()) == ["general", "optics", "particles"]
    assert star["general"].trust_single().to_dict() == {"rlnTomoSubTomosAre2DStacks": 1}
    particles = star["particles"].trust_loop().to_polars()
    assert particles["rlnTomoParticleName"].to_list() == ["TS_01/2"]


def test_restore_star_blocks_nothing_to_restore(tmp_path):
    text = "data_particles\n\nloop_\n_rlnTomoParticleName #1\nTS_01/1\n"
    (tmp_path / "original.star").write_text(text)
    (tmp_path / "filtered.star").write_text(text)

    _impl.restore_star_blocks(tmp_path / "original.star", tmp_path / "filtered.star")

    assert (tmp_path / "filtered.star").read_text() == text  # untouched


def test_check_display(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":1")
    _impl.check_display()

    monkeypatch.delenv("DISPLAY")
    with pytest.raises(RuntimeError, match="display"):
        _impl.check_display()


def test_filter_removes_stale_selection(tmp_path, fake_cryodrgn: FakeCryoDRGN):
    """A selection left by a previous run must not be taken as the new one."""
    workdir = tmp_path / "train"
    workdir.mkdir()
    (workdir / "config.yaml").write_text("{}")
    (workdir / "z.20.pkl").touch()
    (workdir / "analyze.20" / "kmeans20").mkdir(parents=True)
    (workdir / "analyze.20" / "kmeans20" / "labels.pkl").touch()
    _impl.save_indices(tmp_path / "indices.pkl", np.array([0]))
    fake_cryodrgn.filter_selection = None  # window closed without selection

    with pytest.raises(RuntimeError, match="No particles were selected"):
        _impl.cryodrgn_filter(workdir, sel_dir=tmp_path)
    assert not (tmp_path / "indices.pkl").exists()


@pytest.mark.parametrize("bin_dir", [False, True])
def test_run_suppresses_only_stdout(
    monkeypatch: pytest.MonkeyPatch, cryodrgn_bin_dir, bin_dir: bool
):
    """Output is suppressed with or without bin_dir; errors still go to run.err."""
    import subprocess

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(kw))
    _impl._run(
        ["cryodrgn", "downsample"],
        bin_dir=str(cryodrgn_bin_dir) if bin_dir else "",
        suppress_output=True,
    )
    (kwargs,) = calls
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert "stderr" not in kwargs


def test_backproject_indices_with_force_ntilts(tmp_path):
    with pytest.raises(ValueError, match="force_ntilts"):
        _impl.cryodrgn_backproject_voxel(
            tmp_path / "particles.star",
            tmp_path / "ctf.pkl",
            tmp_path / "poses.pkl",
            tmp_path / "out",
            dose_per_tilt=3.0,
            indices_file=tmp_path / "ind.pkl",
        )
