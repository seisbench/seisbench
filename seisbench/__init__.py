import json as _json
import logging as _logging
import os as _os
from pathlib import Path as _Path
from urllib.parse import urljoin as _urljoin

import pkg_resources

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

remote_root = "https://hifis-storage.desy.de:2880/Helmholtz/HelmholtzAI/SeisBench/"

remote_data_root = _urljoin(remote_root, "datasets/")
remote_model_root = _urljoin(remote_root, "models/v3/")

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
__version__ = pkg_resources.get_distribution("seisbench").version

logger = _logging.getLogger("seisbench")
_ch = _logging.StreamHandler()
_ch.setFormatter(
    _logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
logger.addHandler(_ch)
