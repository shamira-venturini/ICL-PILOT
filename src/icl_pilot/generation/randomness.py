"""Run-level random streams; JSON state storage without executable pickles."""
from __future__ import annotations

import base64
import random


def initialise_run_randomness(master_seed: int = 42, *, torch=None) -> None:
    """Call once for a fresh model run, before loading the model. Never on resume."""
    import numpy as np
    if torch is None:
        import torch
    random.seed(master_seed)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(master_seed)


class RandomStreams:
    def __init__(self, *, torch=None):
        if torch is None:
            import torch
        self.torch = torch

    @staticmethod
    def _encode(state):
        return base64.b64encode(state.cpu().numpy().tobytes()).decode("ascii")

    def capture(self) -> dict:
        import numpy as np
        name, keys, position, has_gauss, cached = np.random.get_state()
        cuda = self.torch.cuda
        return {"python": random.getstate(),
                "numpy": {"algorithm": name, "keys": base64.b64encode(keys.astype("<u4").tobytes()).decode("ascii"),
                          "position": position, "has_gauss": has_gauss, "cached_gaussian": cached},
                "torch_cpu": self._encode(self.torch.get_rng_state()),
                "torch_cuda": [self._encode(state) for state in cuda.get_rng_state_all()] if cuda.is_available() else []}

    def restore(self, states: dict) -> None:
        """Restore after model loading; this does not reseed any generator."""
        import numpy as np
        cuda = self.torch.cuda
        count = cuda.device_count() if cuda.is_available() else 0
        if len(states["torch_cuda"]) != count:
            raise ValueError("CUDA device count differs from the saved random streams; checkpoint preserved.")

        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, (tuple, list)) else value

        def tensor(value):
            array = np.frombuffer(base64.b64decode(value, validate=True), dtype=np.uint8).copy()
            return self.torch.tensor(array, dtype=self.torch.uint8, device="cpu")

        cpu, gpu = tensor(states["torch_cpu"]), [tensor(value) for value in states["torch_cuda"]]
        numpy = states["numpy"]
        keys = np.frombuffer(base64.b64decode(numpy["keys"], validate=True), dtype="<u4").copy()
        random.setstate(tuples(states["python"]))
        np.random.set_state((numpy["algorithm"], keys, numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"]))
        self.torch.set_rng_state(cpu)
        if gpu:
            cuda.set_rng_state_all(gpu)
