"""Run the local repository's Ultralytics CLI entrypoint.

Use this instead of the installed `yolo.exe` launcher when training models that
reference custom modules defined in this checkout.
"""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Colab often has wandb preinstalled. Disable it by default because this fork
# passes filesystem paths as Ultralytics project directories, which are invalid
# W&B project names. Set WANDB_DISABLED=false before launch to opt in.
os.environ.setdefault("WANDB_DISABLED", "true")

from ultralytics.cfg import entrypoint


if __name__ == "__main__":
    entrypoint()
