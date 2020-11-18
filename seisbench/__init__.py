import os
from pathlib import Path

# global variable: cache_root
cache_root = os.getenv("SEISBENCH_CACHE_ROOT", Path(Path.home(), ".seisbench"))

# Version number
__version__ = "0.0"
