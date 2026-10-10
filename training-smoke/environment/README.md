# Isolated training environment

Runtime: `/mnt/data/anaconda3/envs/train_mobius_py312`, Python 3.12.14, PyTorch 2.9.1+cu128, Triton 3.5.1, TransformerEngine 2.19.0, Apex CUDA/CPP extensions. Original `train_mobius` packages were retained for baseline replay. `pip-check.txt` records a clean dependency check.

`conda-explicit.txt`, `conda-environment.yaml`, and `pip-freeze.txt` describe the installed environment. The pip export contains local source URLs and Conda build URLs: it is an inventory, not a standalone pip installation recipe. Resolve the local Megatron path to this workspace and the TE/Apex sources to `../vendor/` when rebuilding. Do not install this inventory into the original environment.

`provenance.json` records the frozen source versions, input hashes, and build choices. TE was compiled against CUDA 12.8 with `NVTE_WITH_NCCL_EP=0` and `NVTE_CUDA_ARCHS=90`; Apex was built with CPP and CUDA extensions. The launcher supplies CUDA and NVIDIA wheel library paths. Training was validated in BF16 with FP8 disabled; other modes have not been certified.

The TE source uses Megatron's locked revision `5e52befd5262c06289106338c308079d6adb391f`. Its CUTLASS directory was supplied from an existing local copy reporting version 4.4.2, rather than checked out at the upstream pin `57e3cfb47a2d9e0d46eb6335c3dc411498efa198`. This is a local build, not an exact reproduction of the upstream CI container. The full source/build trees, including compiled extensions, are preserved under ignored `../vendor/`; archive and native-binary checksums are recorded here. Build logs are under `../logs/`.

Baseline replay initially encountered a missing cubin in a shared `/tmp/torchinductor_root` cache. The replay launchers now use workspace-local Python 3.11 Inductor/Triton caches; the new launcher also uses a separate Python 3.12 Inductor cache; no shared cache was deleted. The initial failure log is retained as `baseline-today-cache-error.log`.
