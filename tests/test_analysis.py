"""Reading the outputs of cryoDRGN training (run.log, z, analysis)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from himena_relion_cryodrgn import _analysis

# Lines as written by `cryodrgn train_vae` (see commands/train_vae.py).
RUN_LOG = """\
cryodrgn train_vae particles.star --zdim 8 ...
cryoDRGN 3.5.0
Namespace(particles='particles.star', zdim=8)
# [Train Epoch: 1/3] [1000/2400 particles] gen loss=0.912345, kld=1.234567, beta=0.025000, loss=0.943209
# [Train Epoch: 1/3] [2000/2400 particles] gen loss=0.812345, kld=1.334567, beta=0.025000, loss=0.845709
# =====> Epoch: 1 Average gen loss = 0.851234, KLD = 1.300000, total loss = 0.883734; Finished in 0:01:02
# [Train Epoch: 2/3] [1000/2400 particles] gen loss=0.512345, kld=2.234567, beta=0.025000, loss=0.568209
# =====> Epoch: 2 Average gen loss = 5.5e-05, KLD = 2.100000, total loss = 0.052555; Finished in 0:01:01
"""


def test_parse_train_log(tmp_path):
    (tmp_path / "run.log").write_text(RUN_LOG)

    log = _analysis.parse_train_log(tmp_path / "run.log")

    assert log.epochs == [1, 2]
    assert log.gen_loss == pytest.approx([0.851234, 5.5e-05])  # {:.6} may be 1e-05
    assert log.kld == pytest.approx([1.3, 2.1])
    assert log.total_loss == pytest.approx([0.883734, 0.052555])
    assert log.num_epochs == 3
    assert log.current_epoch == 2
    assert log.last_epoch == 2
    assert not log.finished


def test_parse_train_log_finished_and_prefixed(tmp_path):
    # The console format, in case the file handler ever gets the same formatter.
    prefix = "(INFO) (train_vae.py) (22-Nov-2026 10:00:00) "
    text = "".join(prefix + line + "\n" for line in RUN_LOG.splitlines())
    text += prefix + "Training complete\n" + prefix + "Finished in 0:03:00\n"
    (tmp_path / "run.log").write_text(text)

    log = _analysis.parse_train_log(tmp_path / "run.log")

    assert log.epochs == [1, 2]
    assert log.finished


def test_parse_train_log_resumed(tmp_path):
    # training resumed with --load: epoch 2 is logged again
    resumed = (
        "# =====> Epoch: 2 Average gen loss = 0.4, KLD = 2.0, total loss = 0.45; "
        "Finished in 0:01:00\n"
    )
    (tmp_path / "run.log").write_text(RUN_LOG + resumed)

    log = _analysis.parse_train_log(tmp_path / "run.log")

    assert log.epochs == [1, 2]
    assert log.gen_loss[1] == pytest.approx(0.4)


def test_parse_train_log_missing(tmp_path):
    log = _analysis.parse_train_log(tmp_path / "run.log")
    assert log.epochs == []
    assert log.num_epochs is None
    assert not log.finished


def test_checkpoint_files(tmp_path):
    assert _analysis.is_checkpoint_file(tmp_path / "weights.3.pkl")
    assert _analysis.is_checkpoint_file(tmp_path / "weights.pkl")
    assert not _analysis.is_checkpoint_file(tmp_path / "z.3.pkl")
    assert not _analysis.is_checkpoint_file(tmp_path / "weights.3.pkl.tmp")

    assert _analysis.last_checkpoint_epoch(tmp_path) is None
    for epoch in (1, 2, 10):
        (tmp_path / f"weights.{epoch}.pkl").touch()
    assert _analysis.last_checkpoint_epoch(tmp_path) == 10


def _save_z(path, z):
    with open(path, "wb") as f:
        pickle.dump(z, f)
        pickle.dump(np.zeros_like(z), f)  # z_logvar


def test_load_latent_2d_umap(tmp_path):
    _save_z(tmp_path / "z.5.pkl", np.random.default_rng(0).normal(size=(30, 8)))
    (tmp_path / "analyze.5").mkdir()
    umap = np.random.default_rng(1).normal(size=(30, 2))
    with open(tmp_path / "analyze.5" / "umap.pkl", "wb") as f:
        pickle.dump(umap, f)

    emb = _analysis.load_latent_2d(tmp_path)

    assert emb is not None
    assert (emb.xlabel, emb.ylabel, emb.epoch) == ("UMAP1", "UMAP2", 5)
    np.testing.assert_allclose(emb.xy, umap, rtol=1e-6)


@pytest.mark.parametrize(
    "zdim, labels", [(8, ("PC1", "PC2")), (2, ("z1", "z2")), (1, ("z", ""))]
)
def test_load_latent_2d_without_umap(tmp_path, zdim, labels):
    _save_z(tmp_path / "z.1.pkl", np.zeros((5, zdim)))
    z = np.random.default_rng(0).normal(size=(30, zdim))
    _save_z(tmp_path / "z.2.pkl", z)  # the last epoch is used

    emb = _analysis.load_latent_2d(tmp_path)

    assert emb is not None
    assert emb.xy.shape == (30, 2)
    assert (emb.xlabel, emb.ylabel) == labels
    assert emb.epoch == 2


def test_load_latent_2d_nothing(tmp_path):
    assert _analysis.load_latent_2d(tmp_path) is None
    assert _analysis.load_latent_2d(tmp_path / "missing") is None


def test_load_kmeans_labels(tmp_path):
    for k in (20, 5):
        d = tmp_path / "analyze.3" / f"kmeans{k}"
        d.mkdir(parents=True)
        with open(d / "labels.pkl", "wb") as f:
            pickle.dump(np.arange(10) % k, f)

    assert _analysis.load_kmeans_labels(tmp_path, 3, 20).max() == 9
    assert _analysis.load_kmeans_labels(tmp_path, 3).max() == 4  # smallest k
    assert _analysis.load_kmeans_labels(tmp_path, 3, 7) is None
    assert _analysis.load_kmeans_labels(tmp_path, 4) is None


def _write_mrc(path, value: float = 1.0, voxel_size: float = 5.4):
    import mrcfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.full((5, 5, 5), value, dtype=np.float32))
        mrc.voxel_size = voxel_size


def test_find_volume_series(tmp_path):
    _write_mrc(tmp_path / "backproject" / "backproject.mrc")
    for epoch in (20, 50):
        anlzdir = tmp_path / "train" / f"analyze.{epoch}"
        for i in (1, 2):
            for v in range(1, 12):  # vol_010 and vol_011 must come after vol_009
                _write_mrc(anlzdir / f"pc{i}" / f"vol_{v:03d}.mrc")
        for v in range(1, 6):
            _write_mrc(anlzdir / "kmeans5" / f"vol_{v:03d}.mrc")
        (anlzdir / "pc3").mkdir()  # no volumes (e.g. --skip-vol)

    series = _analysis.find_volume_series(tmp_path)

    assert [s.name for s in series] == [
        "Back-projection (all particles)",
        "PC1 (epoch 50)",
        "PC2 (epoch 50)",
        "k-means 5 (epoch 50)",
        "PC1 (epoch 20)",
        "PC2 (epoch 20)",
        "k-means 5 (epoch 20)",
    ]
    pc1 = series[1]
    assert [p.name for p in pc1.paths] == [f"vol_{v:03d}.mrc" for v in range(1, 12)]


def test_selection_volume_series(tmp_path):
    select, train = tmp_path / "select", tmp_path / "train_job"
    assert _analysis.selection_volume_series(select, train) == []
    _write_mrc(train / "backproject" / "backproject.mrc")
    _write_mrc(select / "backproject" / "backproject.mrc")

    series = _analysis.selection_volume_series(select, train)

    assert [(s.name, s.paths) for s in series] == [
        (
            "Back-projection (selected particles)",
            [select / "backproject" / "backproject.mrc"],
        ),
        (
            "Back-projection (all particles)",
            [train / "backproject" / "backproject.mrc"],
        ),
    ]
    assert len(_analysis.selection_volume_series(select)) == 1  # no Train job


def test_find_volume_series_empty(tmp_path):
    assert _analysis.find_volume_series(tmp_path) == []


def test_read_volume(tmp_path):
    path = tmp_path / "vol_001.mrc"
    _write_mrc(path, value=1.0, voxel_size=4.2)

    img, voxel_size = _analysis.read_volume(path)
    assert img.shape == (5, 5, 5)
    assert voxel_size == pytest.approx(4.2, rel=1e-3)
    assert _analysis.read_volume(path)[0] is img  # cached

    import os

    _write_mrc(path, value=2.0)
    os.utime(path, ns=(0, path.stat().st_mtime_ns + 10**9))
    assert _analysis.read_volume(path)[0][0, 0, 0] == 2.0  # re-read when modified


def test_parse_train_log_continued(tmp_path):
    """Training continued with --load after a finished run is not finished."""
    first = RUN_LOG + "Training complete\nFinished in 0:03:00\n"
    more = (
        "# [Train Epoch: 3/5] [1000/2400 particles] gen loss=0.3, kld=2.0, beta=0.025, "
        "loss=0.35\n"
    )
    (tmp_path / "run.log").write_text(first + more)

    log = _analysis.parse_train_log(tmp_path / "run.log")

    assert not log.finished
    assert log.num_epochs == 5
    assert log.current_epoch == 3


def test_load_kmeans_centers(tmp_path):
    for k, centers in ((20, [3, 1, 4]), (5, [9, 2])):
        d = tmp_path / "analyze.7" / f"kmeans{k}"
        d.mkdir(parents=True)
        np.savetxt(d / "centers_ind.txt", centers, fmt="%d")

    assert _analysis.load_kmeans_centers(tmp_path, 7, 20).tolist() == [3, 1, 4]
    assert _analysis.load_kmeans_centers(tmp_path, 7).tolist() == [9, 2]  # smallest k
    assert _analysis.load_kmeans_centers(tmp_path, 7, 8) is None
    assert _analysis.load_kmeans_centers(tmp_path, 8) is None


def test_find_volume_series_with_traversals(tmp_path):
    import json

    _write_mrc(tmp_path / "backproject" / "backproject.mrc")
    for n, epoch in ((1, 20), (2, 30)):
        d = tmp_path / f"traversal_00{n}"
        for v in range(1, 4):
            _write_mrc(d / f"vol_{v:03d}.mrc")
        (d / "traversal.json").write_text(json.dumps({"epoch": epoch}))
    (tmp_path / "traversal_003").mkdir()  # running: no volumes yet
    for v in range(1, 3):
        _write_mrc(tmp_path / "train" / "analyze.20" / "pc1" / f"vol_{v:03d}.mrc")

    names = [s.name for s in _analysis.find_volume_series(tmp_path)]

    assert names == [
        "Back-projection (all particles)",
        "Graph traversal 2 (epoch 30)",
        "Graph traversal 1 (epoch 20)",
        "PC1 (epoch 20)",
    ]
    assert _analysis.traversal_dirs(tmp_path).keys() == {1, 2, 3}
