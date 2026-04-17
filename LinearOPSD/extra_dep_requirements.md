1. weave
2. immutables
3. vllm
4. 
/data/wyr/LinearOPSD/.venv/lib/python3.10/site-packages/trl/trainer/callbacks.py中的58行左右改成：
with suppress_experimental_warning():
    try:
        from ..experimental.merge_model_callback import MergeModelCallback as _MergeModelCallback
    except Exception:
        class _MergeModelCallback:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "MergeModelCallback requires optional dependencies for mergekit, "
                    "but they are not installed or are incompatible."
                )

    from ..experimental.winrate_callback import WinRateCallback as _

5. uv pip install flash-attn==2.8.3 --no-build-isolation
或者能看具体信息，且有可能快一点的：
uv pip install ninja packaging wheel setuptools
MAX_JOBS=4 uv pip install flash-attn==2.8.3 --no-build-isolation -v
报错：Prepared 1 package in 54m 48s
DEBUG Failed to reflink `/data/wyr/.uv-cache/archive-v0/HlC5mgPZLZpdMaDfh8y_Q/hopper/test_flash_attn.py` to `/data/wyr/LinearOPSD/.venv/lib/python3.10/site-packages/hopper/test_flash_attn.py`: Operation not supported (os error 95), falling back
Installed 1 package in 6ms
 + flash-attn==2.8.3