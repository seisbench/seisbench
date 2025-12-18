import json as _json
import logging as _logging
import os as _os
from pathlib import Path as _Path
from urllib.parse import urljoin as _urljoin
from importlib.metadata import version as _version

__all__ = [
    "cache_root",
    "cache_data_root",
    "cache_model_root",
    "remote_root",
    "remote_data_root",
    "remote_model_root",
    "__version__",
    "config",
]

# global variable: cache_root
cache_root = _Path(
    _os.getenv("SEISBENCH_CACHE_ROOT", _Path(_Path.home(), ".seisbench"))
)

cache_data_root = cache_root / "datasets"
cache_model_root = cache_root / "models" / "v3"
cache_aux_root = cache_root / "auxiliary"

if not cache_root.is_dir():
    cache_root.mkdir(parents=True, exist_ok=True)

_config_path = cache_root / "config.json"
if not _config_path.is_file():
    config = {"dimension_order": "NCW", "component_order": "ZNE"}
    with open(_config_path, "w") as _fconfig:
        _json.dump(config, _fconfig, indent=4, sort_keys=True)
else:
    with open(_config_path, "r") as _fconfig:
        config = _json.load(_fconfig)

# Version number
__version__ = _version("seisbench")

logger = _logging.getLogger("seisbench")
_ch = _logging.StreamHandler()
_ch.setFormatter(
    _logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
logger.addHandler(_ch)


default_remote_root = (
    "https://hifis-storage.desy.de:2880/Helmholtz/HelmholtzAI/SeisBench/"
)
backup_remote_root = "https://seisbench.gfz.de/mirror/"

remote_root = config.get("remote_root", default_remote_root)

remote_data_root = _urljoin(remote_root, "datasets/")
remote_model_root = _urljoin(remote_root, "models/v3/")


def use_backup_repository(backup: str = backup_remote_root):
    """
    Use the backup repository instead of the original one. This is helpful if the main repository is unavailable
    or if your institution/provider blocks access to the main repository. However, this might lead to degraded
    download speeds.

    :param backup: URL of the backup repository. The default is hard-coded.
    """
    logger.warning(
        f"Setting remote root to: {backup}\n"
        f"Please note that this can affect your download speed."
    )
    global remote_root, remote_model_root, remote_data_root
    remote_root = backup

    remote_data_root = _urljoin(backup, "datasets/")
    remote_model_root = _urljoin(backup, "models/v3/")
