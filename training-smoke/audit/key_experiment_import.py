"""Import the hyphenated benchmark script without executing its main."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('key_experiment',Path(__file__).with_name('key-experiment.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
timed=module.timed
