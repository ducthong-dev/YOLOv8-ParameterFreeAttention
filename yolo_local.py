"""Run the local repository's Ultralytics CLI entrypoint.

Use this instead of the installed `yolo.exe` launcher when training models that
reference custom modules defined in this checkout.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.cfg import entrypoint


if __name__ == "__main__":
    entrypoint()
