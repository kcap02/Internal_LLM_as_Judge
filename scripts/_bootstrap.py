"""Make `llm_judge` importable when running scripts directly (no install).

`pip install -e .` makes this a no-op.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Windows DLL-order workaround: in some environments (observed with
# torch 2.5.1 + datasets 4.8.5 on Windows), importing torch BEFORE
# pyarrow/datasets hard-crashes the interpreter (exit code 5, no traceback).
# Importing pyarrow first is harmless everywhere, so pin the safe order here,
# before any llm_judge module pulls in torch.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass
