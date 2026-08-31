from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if sys.path[:1] != [project_root_str]:
    sys.path = [path for path in sys.path if path != project_root_str]
    sys.path.insert(0, project_root_str)

