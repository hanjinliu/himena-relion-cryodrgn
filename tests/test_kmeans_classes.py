"""k-means clusters of the Train job as classes, for Subset selection.

The interactive class selection of himena-relion (relion.select.interactive) is run on
the files written by the Train job, to check that they are read like the files of a
RELION classification job.

Training particles (see conftest), with the k-means labels of the fake cryoDRGN
(np.arange(n) % 20):

    row of z / k-means label   0        1        2        3
    particle                   TS_01/1  TS_01/3  TS_01/5  TS_01/6
    row of particles.star      0        2        4        5
    class number               1        2        3        4    (of 20 classes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from fake_cryodrgn import FakeCryoDRGN
from himena_relion._job_dir import JobDirectory
from himena_relion.hijack.select_interactive import (
    ClassSelectionInput,
    write_selection,
)
from starfile_rs import read_star

from himena_relion_cryodrgn._kmeans_classes import write_kmeans_classes
from himena_relion_cryodrgn.select_particles import TrainJobFiles

if TYPE_CHECKING:
    from conftest import RelionProject

FN_MODEL = "External/job011/kmeans_optimiser.star"


def _particles(star_path):
    return read_star(star_path)["particles"].trust_loop().to_polars()


def _model_classes(project: RelionProject):
    star = read_star(project.job_file("kmeans_model.star"))
    return star["model_classes"].trust_loop().to_polars()


def test_optimiser_star(trained_project: RelionProject):
    star = read_star(trained_project.job_file("kmeans_optimiser.star"))
    # paths relative to the project, as RELION writes them
    assert star["optimiser_general"].trust_single().to_dict() == {
        "rlnModelStarFile": "External/job011/kmeans_model.star",
        "rlnExperimentalDataStarFile": "External/job011/kmeans_data.star",
    }


def test_model_star(trained_project: RelionProject):
    classes = _model_classes(trained_project)

    # the volumes at the cluster centers of the automatic analysis
    refs = classes["rlnReferenceImage"].to_list()
    assert refs == [
        f"External/job011/train/analyze.20/kmeans20/vol_{i:03d}.mrc"
        for i in range(1, 21)
    ]
    assert all((trained_project.root / ref).is_file() for ref in refs)
    distribution = classes["rlnClassDistribution"].to_numpy()
    assert distribution[:4] == pytest.approx([0.25] * 4)
    assert distribution[4:] == pytest.approx(0)


def test_data_star(trained_project: RelionProject):
    data = read_star(trained_project.job_file("kmeans_data.star"))
    original = read_star(trained_project.root / trained_project.particles_star)

    assert list(data.keys()) == list(original.keys())  # general, optics, particles
    assert (
        data["general"].trust_single().to_dict()
        == original["general"].trust_single().to_dict()
    )
    df = data["particles"].trust_loop().to_polars()
    assert dict(zip(df["rlnTomoParticleName"], df["rlnClassNumber"])) == {
        "TS_01/1": 1,
        "TS_01/3": 2,
        "TS_01/5": 3,
        "TS_01/6": 4,
    }
    # the other columns are the rows of particles.star, as they were
    orig_df = original["particles"].trust_loop().to_polars()
    assert df.drop("rlnClassNumber").rows() == orig_df[[0, 2, 4, 5]].rows()


def test_classes_selected_interactively(trained_project: RelionProject):
    """himena-relion's Subset selection reads the clusters and writes particles.star."""
    source = ClassSelectionInput.from_fn_model(FN_MODEL, trained_project.root)

    assert source.num_classes == 20
    assert source.is_3d()  # shown as 3D maps
    assert source.particle_counts().tolist() == [1, 1, 1, 1] + [0] * 16

    select_dir = trained_project.root / "Select" / "job020"
    select_dir.mkdir(parents=True)
    selected = np.zeros(20, dtype=bool)
    selected[[1, 2]] = True  # classes 2 and 3
    result = write_selection(
        JobDirectory(select_dir),
        source,
        selected,
        write_class_averages=False,
        log=lambda _: None,
    )

    assert result.num_particles == 2
    output = select_dir / "particles.star"
    assert list(read_star(output).keys()) == ["general", "optics", "particles"]
    assert _particles(output)["rlnTomoParticleName"].to_list() == ["TS_01/3", "TS_01/5"]


def test_continued_training_updates_classes(
    trained_project: RelionProject, fake_cryodrgn: FakeCryoDRGN
):
    from test_continue import _continue_job

    _continue_job(trained_project, n_epochs=30)

    refs = _model_classes(trained_project)["rlnReferenceImage"].to_list()
    assert refs[0] == "External/job011/train/analyze.30/kmeans20/vol_001.mrc"


def test_not_written_without_volumes(trained_project: RelionProject):
    """e.g. cryodrgn analyze --skip-vol: nothing to show, and old files are removed."""
    trained_project.job_file("train", "analyze.20", "kmeans20", "vol_020.mrc").unlink()
    train = TrainJobFiles.for_job(
        trained_project.root / trained_project.job_dir, trained_project.particles_star
    )

    assert write_kmeans_classes(train, trained_project.root) is None
    for name in ("kmeans_optimiser.star", "kmeans_model.star", "kmeans_data.star"):
        assert not trained_project.job_file(name).exists()
