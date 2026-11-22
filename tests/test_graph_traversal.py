"""Graph Traversal - CryoDRGN Train: a continue job of the Train job.

The continued job.star is made by `make_job_star`, as the scheduler does, and the job
is run with the command line RELION builds from it. Training particles: TS_01/1, /3,
/4(excluded), /5, /6 -> 4 rows in z (see conftest).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena_relion._job_dir import ExternalJobDirectory
from himena_relion.external._other_args import parse_other_args
from relion_command import run_as_relion

from himena_relion_cryodrgn import _analysis
from himena_relion_cryodrgn.graph_traversal import (
    CryoDrgnTrainGraphTraversal,
    parse_anchors,
    prepare_traversal,
)
from himena_relion_cryodrgn.train import CryoDrgnTrainContinue, CryoDrgnTrainJob

if TYPE_CHECKING:
    from conftest import RelionProject

# extracted at 2.7 Å/pix, box 192, trained at box 128
TRAINING_PIXEL_SIZE = 2.7 * 192 / 128


def _train_dir(project: RelionProject) -> Path:
    return project.root / project.job_dir


def _continue(project: RelionProject, job_cls, **params) -> list[str]:
    """Continue the Train job with `job_cls`, as RELION runs it."""
    job_dir = ExternalJobDirectory(_train_dir(project))
    job_star = job_cls(job_dir).make_job_star(**params)
    job_star.to_star_dict().write(job_dir.job_star())
    for name in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE"):
        job_dir.path.joinpath(name).unlink(missing_ok=True)
    return run_as_relion(job_star, f"{project.job_dir.as_posix()}/")


def _traverse(project: RelionProject, **params) -> list[str]:
    return _continue(project, CryoDrgnTrainGraphTraversal, **params)


def test_parse_anchors():
    assert parse_anchors("12 345, 678") == [12, 345, 678]
    assert parse_anchors(" ") == []


def test_suggested_from_train_job():
    from himena_relion._job_class import CONNECTED_JOBS
    from himena_relion.external import pick_job_class

    job_cls = CryoDrgnTrainGraphTraversal
    assert pick_job_class(job_cls.import_path()) is job_cls
    assert (CryoDrgnTrainJob, job_cls) in CONNECTED_JOBS
    # not the same command as continuing the training
    assert job_cls.job_title() == "Graph Traversal"
    assert job_cls.command_id() != CryoDrgnTrainContinue.command_id()


def test_training_parameters_cannot_be_changed():
    names = ["in_mics", "in_parts", "box_size", "lazy", "n_epochs", "dose_per_tilt"]
    names += ["zdim", "enc_layers", "enc_dim", "dec_layers", "dec_dim"]
    names += ["epoch", "anchors", "kmeans", "downsample", "flip"]
    widgets = {name: SimpleNamespace(enabled=True) for name in names}

    CryoDrgnTrainGraphTraversal.setup_widgets(widgets)

    enabled = {name for name, w in widgets.items() if w.enabled}
    assert enabled == {"epoch", "anchors", "kmeans", "downsample", "flip"}


def test_prepare_traversal_kmeans_centers(trained_project: RelionProject):
    plan = prepare_traversal(_train_dir(trained_project))

    assert plan.epoch == 20  # the last epoch
    assert plan.anchors == [0, 1, 2, 3]  # kmeans20/centers_ind.txt
    assert plan.anchors_from == "k-means centers"
    assert plan.z_file == trained_project.job_file("train", "z.20.pkl")
    assert plan.weights == trained_project.job_file("train", "weights.20.pkl")
    assert plan.apix == pytest.approx(TRAINING_PIXEL_SIZE)  # not 1 Å
    assert plan.downsample is None


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"downsample": 63}, ValueError, "downsample"),
        ({"downsample": 128}, ValueError, "downsample"),
        ({"anchors": "2"}, ValueError, "two anchors"),
        ({"anchors": "3, a"}, ValueError, "integers"),
        ({"epoch": 99}, FileNotFoundError, "z.99.pkl"),
    ],
)
def test_prepare_traversal_errors(
    trained_project: RelionProject, kwargs: dict, error: type, match: str
):
    with pytest.raises(error, match=match):
        prepare_traversal(_train_dir(trained_project), **kwargs)


def test_traversal(trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN):
    argv = _traverse(trained_project)

    assert CryoDrgnTrainGraphTraversal.import_path() in argv
    assert fake_cryodrgn.subcommands == ["graph_traversal", "eval_vol"]
    out = trained_project.job_file("traversal_001")
    gt = fake_cryodrgn.parsed("graph_traversal")
    assert Path(gt.zfile) == trained_project.job_file("train", "z.20.pkl")
    assert gt.anchors == ["0", "1", "2", "3"]  # k-means centers
    assert Path(gt.outtxt) == out / "z_path.txt"
    ev = fake_cryodrgn.parsed("eval_vol")
    assert Path(ev.weights) == trained_project.job_file("train", "weights.20.pkl")
    assert Path(ev.config) == trained_project.job_file("train", "config.yaml")
    assert Path(ev.zfile) == out / "z_path.txt"
    assert Path(ev.o) == out
    assert ev.Apix == pytest.approx(TRAINING_PIXEL_SIZE, rel=1e-4)

    assert trained_project.job_file("RELION_JOB_EXIT_SUCCESS").exists()
    assert len(_analysis.volume_files(out)) == 7  # one per particle on the path
    info = json.loads((out / "traversal.json").read_text())
    assert (info["epoch"], info["anchors"], info["anchors_from"]) == (
        20,
        [0, 1, 2, 3],
        "k-means centers",
    )
    # the outputs of the training are untouched
    assert trained_project.job_file("weights.pkl").exists()
    assert trained_project.job_file("particles.star").exists()
    assert trained_project.job_file("backproject", "backproject.mrc").exists()


def test_each_traversal_in_a_new_directory(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    _traverse(trained_project)
    _traverse(trained_project, anchors="3, 0", downsample=64, flip=True)

    first, second = (trained_project.job_file(f"traversal_00{i}") for i in (1, 2))
    assert len(_analysis.volume_files(first)) == 7  # kept
    assert len(_analysis.volume_files(second)) == 3
    gt = [c for c in fake_cryodrgn.calls if c[1] == "graph_traversal"][-1]
    assert gt[gt.index("--anchors") + 1 : gt.index("--anchors") + 3] == ["3", "0"]
    ev = [c for c in fake_cryodrgn.calls if c[1] == "eval_vol"][-1]
    assert ev[ev.index("-d") + 1] == "64"
    assert float(ev[ev.index("--Apix") + 1]) == pytest.approx(
        TRAINING_PIXEL_SIZE * 128 / 64, rel=1e-4
    )
    assert "--flip" in ev


def test_anchors_are_written_without_spaces(trained_project: RelionProject):
    """RELION passes values unquoted: "3, 0" would reach the job as "3,"."""
    job_dir = ExternalJobDirectory(_train_dir(trained_project))
    job_star = CryoDrgnTrainGraphTraversal(job_dir).make_job_star(anchors=" 3,  0 12")
    opts = dict(
        zip(job_star.joboptions_values.variable, job_star.joboptions_values.value)
    )
    params = {
        opts[f"param{i}_label"]: opts[f"param{i}_value"]
        for i in range(1, 11)
        if opts[f"param{i}_label"]
    }
    # beyond the 10 slots: in other_args ("Additional arguments")
    params.update(parse_other_args(opts["other_args"]))
    assert params["anchors"] == "3,0,12"
    assert job_star.job.job_is_continue == 1


def test_invalid_anchors_are_rejected_before_scheduling():
    with pytest.raises(ValueError, match="integers"):
        CryoDrgnTrainGraphTraversal.prerun_check(anchors="3, a")


def test_traversal_errors_before_running(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    with pytest.raises(ValueError, match="downsample"):
        _traverse(trained_project, downsample=63)
    assert fake_cryodrgn.calls == []
    assert trained_project.job_file("RELION_JOB_EXIT_FAILURE").exists()
    assert not trained_project.job_file("traversal_001").exists()


def test_traversal_without_kmeans(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    shutil.rmtree(trained_project.job_file("train", "analyze.20"))
    with pytest.raises(RuntimeError, match="No k-means result"):
        _traverse(trained_project)
    _traverse(trained_project, anchors="0,3")  # anchors can still be given
    assert len(_analysis.volume_files(trained_project.job_file("traversal_001"))) == 3


def test_continue_training_after_traversal(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    """The parameters of the traversal do not disturb continuing the training."""
    _traverse(trained_project, anchors="0,3")
    fake_cryodrgn.calls.clear()

    _continue(trained_project, CryoDrgnTrainContinue, n_epochs=30)

    assert fake_cryodrgn.subcommands == ["train_vae"]
    assert fake_cryodrgn.parsed("train_vae").num_epochs == 30
    assert trained_project.job_file("traversal_001").is_dir()
