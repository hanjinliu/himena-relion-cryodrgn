import os
from typing import Annotated

from himena_relion.external import RelionExternalJob

from himena_relion_cryodrgn import _configs, _impl

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
    """Base class of the cryoDRGN jobs: where the cryoDRGN commands are.

    Subclasses must have a `cryodrgn_bin_dir: CRYODRGN_BIN_DIR = ""` parameter in
    `run()`. The directory is taken from the settings when the job is created and
    recorded in job.star, because the job process cannot read the settings.
    """

    @classmethod
    def _resolve_bin_dir(cls, kwargs: dict) -> str:
        value = kwargs.get("cryodrgn_bin_dir") or ""
        return os.path.expanduser(str(value).strip()) or _configs.get_bin_dir()

    @classmethod
    def prerun_check(cls, **kwargs) -> None:
        super().prerun_check(**kwargs)
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
