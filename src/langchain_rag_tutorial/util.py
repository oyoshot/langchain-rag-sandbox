from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
STATE_OF_THE_UNION_TXT = DATA_DIR / "state_of_the_union.txt"

BUILD_DIR = PROJECT_ROOT / "build"
VECTOR_DIR = BUILD_DIR / "vector"
