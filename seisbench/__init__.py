import os as _os
from pathlib import Path as _Path
import json as _json
import logging as _logging

__all__ = ["cache_root", "__version__", "config"]

# global variable: cache_root
cache_root = _Path(
    _os.getenv("SEISBENCH_CACHE_ROOT", _Path(_Path.home(), ".seisbench"))
)

remote_root = "https://dcache-demo.desy.de:2443/Helmholtz/HelmholtzAI/SeisBench/"

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
__version__ = "0.2.8"

logger = _logging.getLogger("seisbench")
_ch = _logging.StreamHandler()
_ch.setFormatter(
    _logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
logger.addHandler(_ch)
