"""Import the existing hyphenated audit script without executing its main."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('operator_precision', Path(__file__).with_name('operator-precision.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
key_current = module.key_current
key_formula = module.key_formula
