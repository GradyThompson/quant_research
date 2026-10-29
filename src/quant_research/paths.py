from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Project directories
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data directories
DEV_DATA_DIR = DATA_DIR / "dev"
RESEARCH_DATA_DIR = DATA_DIR / "research"
