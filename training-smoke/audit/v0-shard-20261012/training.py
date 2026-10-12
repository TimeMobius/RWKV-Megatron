"""Run unmodified production training with actual scalar/data/replica capture."""
import os
import runpy
from pathlib import Path

PATH=Path(__file__).resolve().parent
if os.environ.get('V0_PAYLOAD_BEFORE')=='1':
    from legacy import install
    install()
# Redirect only audit output constants; training and loss math remain production.
script=(PATH.parent/'multigpu-20261012/training.py').read_text()
script=script.replace("OUT=ROOT/'training-smoke/audit/multigpu-20261012'", "OUT=ROOT/'training-smoke/audit/v0-shard-20261012'")
script=script.replace("active.update(tp_replicated_parameters_exact", "active.update(cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20,cuda_allocated_mib=torch.cuda.memory_allocated()/2**20)\n        active.update(tp_replicated_parameters_exact")
exec(compile(script,str(PATH.parent/'multigpu-20261012/training.py'),'exec'),{'__name__':'__main__','__file__':str(PATH.parent/'multigpu-20261012/training.py')})
