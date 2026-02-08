import sys
from pathlib import Path

# Add backend/ to sys.path so bare imports (e.g. `from vector_store import ...`) resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
