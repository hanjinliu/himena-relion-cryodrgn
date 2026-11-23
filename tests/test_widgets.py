"""Job widgets of the CryoDRGN jobs (Qt, offscreen)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena_relion._job_dir import ExternalJobDirectory
from himena_relion.external import run_function
from qtpy import QtWidgets as QtW

from himena_relion_cryodrgn import _analysis
from himena_relion_cryodrgn.select_particles import CryoDrgnSelectParticlesJob
from himena_relion_cryodrgn.train import CryoDrgnTrainJob
from himena_relion_cryodrgn.widgets import (
    QCryoDrgnPlotCanvas,
    QCryoDrgnSelectViewer,
    QCryoDrgnTrainViewer,
)

if TYPE_CHECKING:
    from conftest import RelionProject


@pytest.fixture(autouse=True)
def _sync_viewers():
    """Render synchronously, as in the tests of himena-relion."""
    from himena_relion._impl_objects import set_is_testing
    from himena_relion._widgets._view_nd import QViewer

    set_is_testing(True)
    QViewer._always_force_sync = True


def _lines(canvas: QCryoDrgnPlotCanvas):
    return canvas.figure.axes[0].lines


def _scatters(canvas: QCryoDrgnPlotCanvas):
    return canvas.figure.axes[0].collections


def _train_widget(qtbot, project: RelionProject) -> QCryoDrgnTrainViewer:
    job_dir = ExternalJobDirectory(project.root / project.job_dir)
    widget = CryoDrgnTrainJob(job_dir).provide_widget(job_dir)
    assert isinstance(widget, QCryoDrgnTrainViewer)
    qtbot.addWidget(widget)
    widget.initialize(job_dir)
    return widget


def _epoch_lines(epoch: int, num_epochs: int = 20) -> str:
    return (
        f"# [Train Epoch: {epoch}/{num_epochs}] [100/100 particles] gen loss=0.5, "
        "kld=1.0, beta=0.025000, loss=0.525\n"
        f"# =====> Epoch: {epoch} Average gen loss = {1 / epoch:.6}, KLD = 1.000000, "
        f"total loss = {1 / epoch + 0.025:.6f}; Finished in 0:00:01\n"
    )


def test_train_widget_before_training(qtbot, relion_project: RelionProject):
    widget = _train_widget(qtbot, relion_project)

    assert widget._status.text() == "Preparing particles for training..."
    assert len(_lines(widget._loss_canvas)) == 0
    assert len(_scatters(widget._latent_canvas)) == 0
    assert not widget._volumes._viewer.has_image


def test_train_widget_after_training(qtbot, trained_project: RelionProject):
    widget = _train_widget(qtbot, trained_project)

    assert widget._status.text() == "Training complete (20 epochs)."
    total, gen = _lines(widget._loss_canvas)
    assert len(total.get_xdata()) == len(gen.get_xdata()) == 20
    (kld,) = _lines(widget._kld_canvas)
    assert list(kld.get_xdata()) == list(range(1, 21))
    # UMAP of the automatic analysis, one scatter per k-means cluster
    assert "epoch 20, UMAP, 4 k-means clusters" in widget._latent_title.text()
    assert len(_scatters(widget._latent_canvas)) == 4
    n_points = sum(len(c.get_offsets()) for c in _scatters(widget._latent_canvas))
    assert n_points == 4  # training particles (TS_01/2 and TS_01/4 excluded)
    # the class numbers of Subset selection (k-means label + 1 = volume number)
    texts = widget._latent_canvas.figure.axes[0].texts
    assert sorted(t.get_text() for t in texts) == ["1", "2", "3", "4"]
    assert widget._volumes._viewer.has_image  # back-projection
    widget.widget_added_callback()


def test_learning_curve_updates_with_checkpoints(
    qtbot, relion_project: RelionProject, monkeypatch: pytest.MonkeyPatch
):
    widget = _train_widget(qtbot, relion_project)
    n_parsed = []
    parse = _analysis.parse_train_log
    monkeypatch.setattr(
        _analysis, "parse_train_log", lambda p: n_parsed.append(p) or parse(p)
    )
    train = relion_project.job_file("train")
    train.mkdir()
    run_log = train / "run.log"
    run_log.write_text("".join(_epoch_lines(e) for e in (1, 2, 3)))
    (train / "weights.3.pkl").touch()

    widget.on_job_files_updated(widget._job_dir, [train / "weights.3.pkl"])

    assert len(_lines(widget._loss_canvas)[0].get_xdata()) == 3
    assert widget._status.text().startswith("Epoch 3 / 20 finished.")
    # all the epochs to train: n_epochs is not in this job.star, but in run.log
    for canvas in (widget._loss_canvas, widget._kld_canvas):
        assert canvas.figure.axes[0].get_xlim() == pytest.approx((0.85, 20.15))

    # run.log changes during an epoch: the plot is not updated
    with open(run_log, "a") as f:
        f.write(_epoch_lines(4))
    widget.on_job_files_updated(widget._job_dir, [run_log, train / "z.3.pkl.tmp"])
    assert len(n_parsed) == 1
    assert len(_lines(widget._loss_canvas)[0].get_xdata()) == 3

    # ... until the checkpoint of the epoch is written
    (train / "weights.4.pkl").touch()
    widget.on_job_files_updated(widget._job_dir, [str(train / "weights.4.pkl")])
    assert len(n_parsed) == 2
    assert len(_lines(widget._loss_canvas)[0].get_xdata()) == 4
    assert widget._status.text().startswith("Epoch 4 / 20 finished.")


def test_train_widget_shows_latent_during_training(
    qtbot, trained_project: RelionProject
):
    """Before the analysis exists, the PCA of the last z is shown."""
    import shutil

    shutil.rmtree(trained_project.job_file("train", "analyze.20"))
    widget = _train_widget(qtbot, trained_project)

    assert "epoch 20, PCA of z" in widget._latent_title.text()
    assert len(_scatters(widget._latent_canvas)) == 1


def test_train_widget_latent_of_earlier_epoch(qtbot, trained_project: RelionProject):
    widget = _train_widget(qtbot, trained_project)
    choice = widget._latent_epoch
    assert not choice.isHidden()
    assert choice.choices() == list(range(1, 21))
    assert choice.value() == 20

    # no analysis of epoch 5: the PCA of z.5.pkl, without the k-means clusters
    choice.setValue(5)
    assert "epoch 5, PCA of z" in widget._latent_title.text()
    (scatter,) = _scatters(widget._latent_canvas)
    assert len(scatter.get_offsets()) == 4

    choice.setValue(20)
    assert "epoch 20, UMAP, 4 k-means clusters" in widget._latent_title.text()
    assert len(_scatters(widget._latent_canvas)) == 4


def test_epoch_choice_steps_through_checkpoints(qtbot, trained_project: RelionProject):
    """Only the epochs with z.{epoch}.pkl, e.g. with `--checkpoint 5`."""
    for epoch in range(1, 21):
        if epoch % 5:
            trained_project.job_file("train", f"z.{epoch}.pkl").unlink()
    widget = _train_widget(qtbot, trained_project)
    choice = widget._latent_epoch
    assert choice.choices() == [5, 10, 15, 20]

    choice._left_btn.click()
    assert choice.value() == 15
    assert "epoch 15, PCA of z" in widget._latent_title.text()
    choice._left_btn.click()
    choice._left_btn.click()
    assert choice.value() == 5
    choice.setValue(12)  # the next epoch with z.{epoch}.pkl, from the current one
    assert choice.value() == 10
    assert "epoch 10, PCA of z" in widget._latent_title.text()


def test_epoch_choice_follows_training(qtbot, relion_project: RelionProject):
    from fake_cryodrgn import _save_z

    widget = _train_widget(qtbot, relion_project)
    choice = widget._latent_epoch
    assert choice.isHidden()
    train = relion_project.job_file("train")
    train.mkdir()
    rng = np.random.default_rng(0)
    plotted = []
    plot_latent = widget._latent_canvas.plot_latent
    widget._latent_canvas.plot_latent = lambda emb, **kw: (
        plotted.append(emb.epoch) or plot_latent(emb, **kw)
    )

    def _write_epoch(epoch: int):
        _save_z(train / f"z.{epoch}.pkl", rng.normal(size=(4, 8)))
        plotted.clear()
        widget.on_job_files_updated(widget._job_dir, [train / f"z.{epoch}.pkl"])

    _write_epoch(1)
    _write_epoch(2)
    assert not choice.isHidden()
    assert choice.value() == 2  # the last epoch is followed
    assert "epoch 2, PCA of z" in widget._latent_title.text()
    assert plotted == [2]  # once

    choice.setValue(1)
    _write_epoch(3)
    assert choice.choices() == [1, 2, 3]
    assert choice.value() == 1  # the chosen epoch is kept
    assert "epoch 1, PCA of z" in widget._latent_title.text()
    assert plotted == [1]

    choice._right_btn.click()
    choice._right_btn.click()
    assert choice.value() == 3
    _write_epoch(4)
    assert choice.value() == 4
    assert "epoch 4, PCA of z" in widget._latent_title.text()
    assert plotted == [4]


def _write_select_job_star(project: RelionProject) -> None:
    """RELION writes job.star when the job is created."""
    CryoDrgnSelectParticlesJob.prep_job_star(
        train_weights=f"{project.job_dir.as_posix()}/weights.pkl", epoch=-1, kmeans=-1
    ).to_star_dict().write(project.select_job_file("job.star"))


def _select_widget(qtbot, project: RelionProject) -> QCryoDrgnSelectViewer:
    job_dir = ExternalJobDirectory(project.root / project.select_job_dir)
    widget = CryoDrgnSelectParticlesJob(job_dir).provide_widget(job_dir)
    assert isinstance(widget, QCryoDrgnSelectViewer)
    qtbot.addWidget(widget)
    widget.initialize(job_dir)
    return widget


def test_select_widget_waiting(qtbot, trained_project: RelionProject):
    _write_select_job_star(trained_project)
    widget = _select_widget(qtbot, trained_project)

    assert widget._summary.text().startswith("Waiting for the selection")
    assert len(_scatters(widget._latent_canvas)) == 1  # all training particles
    # the back-projection of the Train job, before the selection is back-projected
    assert _volume_sources(widget) == ["Back-projection (all particles)"]
    assert widget._volumes._viewer.has_image


def test_select_widget_after_selection(
    qtbot,
    trained_project: RelionProject,
    fake_cryodrgn: FakeCryoDRGN,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_select_job_star(trained_project)
    widget = _select_widget(qtbot, trained_project)
    fake_cryodrgn.filter_selection = [1, 2]  # points of TS_01/3 and TS_01/5
    run_function(
        [
            CryoDrgnSelectParticlesJob.import_path(),
            "--o", f"{trained_project.select_job_dir.as_posix()}/",
            "--train_weights", f"{trained_project.job_dir.as_posix()}/weights.pkl",
        ]
    )  # fmt: skip

    # himena does not run in the project directory: paths must not depend on cwd
    monkeypatch.chdir(Path(trained_project.root).parent)
    widget.on_job_files_updated(
        widget._job_dir,
        [
            trained_project.select_job_file("indices.pkl"),
            trained_project.select_job_file("particles.star"),
            trained_project.select_job_file("backproject", "backproject.mrc"),
        ],
    )

    assert widget._summary.text() == "Selected: <b>2</b>/6 particles"
    not_selected, selected = _scatters(widget._latent_canvas)
    # the lassoed points, not just any two points
    umap = _analysis.load_latent_2d(trained_project.job_file("train")).xy
    np.testing.assert_allclose(selected.get_offsets(), umap[[1, 2]], rtol=1e-5)
    np.testing.assert_allclose(not_selected.get_offsets(), umap[[0, 3]], rtol=1e-5)
    widget.widget_added_callback()
    # the back-projection of the selection is shown, to compare with all particles
    assert _volume_sources(widget) == [
        "Back-projection (selected particles)",
        "Back-projection (all particles)",
    ]
    assert widget._volumes._source.currentIndex() == 0
    assert widget._volumes._info.text().startswith("backproject.mrc")
    assert widget._volumes._index.isHidden()  # a single volume each


def test_select_widget_without_train_job(qtbot, relion_project: RelionProject):
    CryoDrgnSelectParticlesJob.prep_job_star(
        train_weights="External/job099/weights.pkl"
    ).to_star_dict().write(relion_project.select_job_file("job.star"))
    widget = _select_widget(qtbot, relion_project)

    assert widget._summary.text().startswith("CryoDRGN Train job not found")
    assert _volume_sources(widget) == []


def _volume_sources(widget: QCryoDrgnTrainViewer | QCryoDrgnSelectViewer) -> list[str]:
    combo = widget._volumes._source
    return [combo.itemText(i) for i in range(combo.count())]


def test_volume_viewer_reads_volumes_lazily(
    qtbot, trained_project: RelionProject, monkeypatch: pytest.MonkeyPatch
):
    reads = []
    read_volume = _analysis.read_volume
    monkeypatch.setattr(
        _analysis, "read_volume", lambda p: reads.append(Path(p).name) or read_volume(p)
    )
    widget = _train_widget(qtbot, trained_project)

    assert _volume_sources(widget) == [
        "Back-projection (all particles)",
        "PC1 (epoch 20)",
        "PC2 (epoch 20)",
        "k-means 20 (epoch 20)",
    ]
    assert reads == ["backproject.mrc"]  # not the 40 volumes of the analysis
    assert widget._volumes._index.isHidden()  # a single volume

    widget._volumes._source.setCurrentIndex(1)  # PC1
    assert not widget._volumes._index.isHidden()
    assert widget._volumes._index.maximum() == 10
    assert reads == ["backproject.mrc", "vol_001.mrc"]

    # a slider without label: the name of the volume is in the info line
    assert isinstance(widget._volumes._index, QtW.QSlider)
    widget._volumes._index.setValue(3)
    assert reads[-1] == "vol_003.mrc"
    assert len(reads) == 3
    assert widget._volumes._info.text().startswith("vol_003.mrc  (9x9x9, 5.4 Å/pix)")
    assert widget._volumes._viewer.has_image

    widget._volumes._source.setCurrentIndex(3)  # k-means
    assert widget._volumes._index.maximum() == 20


def test_volume_viewer_keeps_selection_on_update(qtbot, trained_project: RelionProject):
    widget = _train_widget(qtbot, trained_project)
    widget._volumes._source.setCurrentIndex(2)  # PC2
    widget._volumes._index.setValue(7)

    # e.g. `cryodrgn analyze` wrote another volume
    vol = trained_project.job_file("train", "analyze.20", "pc2", "vol_007.mrc")
    widget.on_job_files_updated(widget._job_dir, [vol])

    assert widget._volumes._source.currentText() == "PC2 (epoch 20)"
    assert widget._volumes._index.value() == 7
    assert widget._volumes._info.text().startswith("vol_007.mrc")


def test_volume_viewer_before_training(qtbot, relion_project: RelionProject):
    widget = _train_widget(qtbot, relion_project)
    assert _volume_sources(widget) == []
    assert widget._volumes._info.text() == ""
    assert not widget._volumes._viewer.has_image


def test_train_widget_after_continue(
    qtbot, trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    from test_continue import _continue_job

    _continue_job(trained_project, n_epochs=30)
    widget = _train_widget(qtbot, trained_project)

    assert widget._status.text() == "Training complete (30 epochs)."
    total, _ = _lines(widget._loss_canvas)
    assert len(total.get_xdata()) == 30
    # n_epochs of job.star, the total number of epochs
    assert widget._loss_canvas.figure.axes[0].get_xlim() == pytest.approx((0.85, 30.15))
    assert "epoch 30" in widget._latent_title.text()
    sources = _volume_sources(widget)
    assert sources[1] == "PC1 (epoch 30)"  # the latest analysis first
    assert "PC1 (epoch 20)" in sources


def test_volume_viewer_shows_graph_traversals(
    qtbot, trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    from test_graph_traversal import _traverse

    widget = _train_widget(qtbot, trained_project)
    assert not any(s.startswith("Graph traversal") for s in _volume_sources(widget))

    _traverse(trained_project, anchors="0,3")  # 3 volumes in traversal_001
    vol = trained_project.job_file("traversal_001", "vol_003.mrc")
    widget.on_job_files_updated(widget._job_dir, [vol])

    sources = _volume_sources(widget)
    assert sources[:2] == [
        "Back-projection (all particles)",
        "Graph traversal 1 (epoch 20)",
    ]
    widget._volumes._source.setCurrentIndex(1)
    assert widget._volumes._index.maximum() == 3
    widget._volumes._index.setValue(2)
    assert widget._volumes._info.text().startswith("vol_002.mrc")
    assert widget._volumes._viewer.has_image
