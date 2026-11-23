import os
from typing import Annotated

from himena_relion._widgets._magicgui import GpuIdEdit
from himena_relion.external import RelionExternalJob

from himena_relion_cryodrgn import _configs, _impl

GPU_IDS = Annotated[
    str,
    {
        "label": "GPU IDs to use",
        "widget_type": GpuIdEdit,
        "tooltip": (
            "GPUs to run cryoDRGN on, separated by commas (e.g. 0 or 0,1), numbered as\n"
            "in RELION. With more than one GPU, the training is parallelized over them\n"
            "(cryodrgn train_vae --multigpu); the other commands use the first one. If\n"
            "empty, cryoDRGN uses the first GPU visible to the job."
        ),
        "group": "Running",
    },
]
CRYODRGN_BIN_DIR = Annotated[
    str,
    {
        "label": "cryoDRGN bin directory",
        "tooltip": (
            "Directory with the cryodrgn and cryodrgn_utils commands. If empty, the\n"
            "directory in the settings (CryoDRGN) is used, or PATH if it is not set."
        ),
        "group": "Running",
    },
]


class _CryoDrgnJob(RelionExternalJob):
    """Base class of the cryoDRGN jobs: where the cryoDRGN commands are, and on which
    GPUs they run.

    Subclasses must have a `cryodrgn_bin_dir: CRYODRGN_BIN_DIR = ""` parameter in
    `run()`. The directory is taken from the settings when the job is created and
    recorded in job.star, because the job process cannot read the settings.
    Subclasses running cryoDRGN commands on the GPU also have a
    `gpu_ids: GPU_IDS = ""` parameter, parsed by `_gpu_ids()`.
    """

    @classmethod
    def _resolve_bin_dir(cls, kwargs: dict) -> str:
        value = kwargs.get("cryodrgn_bin_dir") or ""
        return os.path.expanduser(str(value).strip()) or _configs.get_bin_dir()

    def _gpu_ids(self, value: str) -> list[int]:
        """The GPU IDs to pass to the cryoDRGN commands (see `_impl._run`).

        The GPUs are checked against CUDA_VISIBLE_DEVICES of the job process before
        any command runs, and logged.
        """
        gpu_ids = _impl.parse_gpu_ids(value)
        if gpu_ids:
            visible = os.environ.get(_impl.CUDA_VISIBLE_DEVICES)
            devices = _impl.cuda_visible_devices(gpu_ids, visible)
            self.console.log(
                f"cryoDRGN runs on GPU {','.join(map(str, gpu_ids))} "
                f"({_impl.CUDA_VISIBLE_DEVICES}={devices})."
            )
        return gpu_ids

    @classmethod
    def prerun_check(cls, **kwargs) -> None:
        super().prerun_check(**kwargs)
        _impl.parse_gpu_ids(str(kwargs.get("gpu_ids", "")))
        bin_dir = cls._resolve_bin_dir(kwargs)
        if any(c.isspace() for c in bin_dir):
            # RELION passes the parameters of External jobs to the command unquoted.
            raise ValueError(
                f"The cryoDRGN bin directory must not contain spaces: {bin_dir!r}"
            )
        for name in (_impl.CRYODRGN, _impl.CRYODRGN_UTILS):
            if _impl.find_command(name, bin_dir) is None:
                where = bin_dir or "PATH"
                raise ValueError(
                    f"`{name}` was not found in {where}. Set the directory of the "
                    "cryoDRGN commands in the settings (CryoDRGN)."
                )

    @classmethod
    def prep_job_star(cls, **kwargs):
        if bin_dir := cls._resolve_bin_dir(kwargs):
            kwargs["cryodrgn_bin_dir"] = bin_dir
        # Empty strings are not written to job.star, so that the default ("") is used:
        # RELION passes a parameter with an empty value as a bare flag (`--name`),
        # which himena-relion reads as True.
        kwargs = {
            k: v for k, v in kwargs.items() if not (isinstance(v, str) and v == "")
        }
        return super().prep_job_star(**kwargs)
