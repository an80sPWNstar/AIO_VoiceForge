"""Print the torch device string for a GPU matched by name.

The start script needs the CUDA index of one specific card, and torch's
device order is not nvidia-smi's, so the index cannot be a constant: on this
machine nvidia-smi calls the 5060 Ti card 1 while torch calls it cuda:2.
This lives in its own file rather than as a python -c one-liner because batch
mangles nested quotes.

Prints "cuda:N" for the first match and exits 0. Prints nothing and exits 1
when torch is missing, CUDA is unavailable, or no card matches -- the caller
treats empty output as "use the fallback".
"""

import sys

DEFAULT_FRAGMENT = "5060"


def main() -> int:
    fragment = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FRAGMENT).lower()
    # Any failure prints nothing: a traceback would land in the caller's
    # for /f capture and be read back as a device string.
    try:
        import torch

        for index in range(torch.cuda.device_count()):
            if fragment in torch.cuda.get_device_name(index).lower():
                print(f"cuda:{index}")
                return 0
    except Exception:
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
