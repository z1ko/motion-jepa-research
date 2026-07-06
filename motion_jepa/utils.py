
import torch as T

def ema_linear_scheduler(max_steps: int, global_step: int, ema_momentun: float) -> float:
    if max_steps is None or max_steps <= 0:
        raise ValueError("Linear EMA scheduling requires Trainer(max_steps=...).")

    progress = min(global_step / max_steps, 1.0)
    ema = ema_momentun + progress * (1.0 - ema_momentun)
    return ema
    