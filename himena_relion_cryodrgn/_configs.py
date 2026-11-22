import os
from dataclasses import dataclass

from himena.plugins import config_field, get_config, register_config

PLUGIN_ID = "himena-relion-cryodrgn"


@dataclass
class CryoDrgnConfig:
    # config_field returns a dataclass field, which pyright cannot type as str.
    bin_dir: str = config_field(  # pyright: ignore[reportAssignmentType]
        default="",
        label="cryoDRGN <code>bin</code> Directory",
        tooltip=(
            "Directory with the cryodrgn and cryodrgn_utils commands, such as\n"
            "~/miniforge3/envs/cryodrgn/bin, when cryoDRGN is installed in another\n"
            "environment. Leave it empty if they are in your PATH."
        ),
    )


register_config(PLUGIN_ID, "CryoDRGN", CryoDrgnConfig())


def get_bin_dir() -> str:
    """The configured directory of the cryoDRGN commands ("" means PATH).

    The config is only available in the himena GUI process; "" is returned elsewhere,
    e.g. in the process that runs a job. Jobs therefore record the directory in
    job.star when they are created.
    """
    try:
        config = get_config(CryoDrgnConfig, PLUGIN_ID)
    except (StopIteration, LookupError):  # no himena main window in this process
        return ""
    if config is None:
        return ""
    return os.path.expanduser(config.bin_dir.strip())
