"""Set GPU visibility before importing torch or any experiment modules."""
import argparse
import os
import sys

if __name__ == "__main__":
    gpu_parser = argparse.ArgumentParser(add_help=False)
    gpu_parser.add_argument("--gpu", help="physical CUDA GPU index, e.g. 0 or 1")
    gpu, remaining = gpu_parser.parse_known_args()
    if gpu.gpu is not None:
        if not gpu.gpu.isdigit():
            raise ValueError("--gpu must be a nonnegative integer")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu.gpu
    from ..launcher import main
    sys.exit(main(remaining))
