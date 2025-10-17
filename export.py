import sys
from pathlib import Path

sys.path.append(str(Path(".") / "implementations/cgan"))

import torch
from cgan import Generator

if __name__ == "__main__":
    x = torch.randn(100, 100)
    labels = torch.arange(100)
    model = Generator()
    try:
        torch.export.export(model, (x, labels))
        print("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print("[JIT] torch.export failed.")
        raise e
