import time
from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class TrainMonitor:
    """Track and report training progress.

    Parameters
    ----------
    log_every
        Print progress every this many environment steps.
    ret_window
        Rolling window (episodes) for return/length stats.
    loss_window
        Rolling window (updates) for loss stats.

    """
    log_every: int = 1000
    ret_window: int = 50
    loss_window: int = 200
    t0: float = field(default_factory=time.time)
    last_log_t: float = field(default_factory=time.time)
    ep_rets: List[float] = field(default_factory=list)
    ep_lens: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    last_s: int = -1
    last_a: int = -1
    last_s2: int = -1

    def add_episode(self, ret: float, length: int) -> None:
        """Record a finished episode."""
        self.ep_rets.append(float(ret))
        self.ep_lens.append(int(length))

    def add_loss(self, loss: float) -> None:
        """Record a loss value from an update."""
        self.losses.append(float(loss))

    def _mean_last(self, x: List[float], k: int) -> float:
        if len(x) == 0:
            return float("nan")
        k = min(int(k), len(x))
        return float(np.mean(x[-k:]))

    def _mean_last_int(self, x: List[int], k: int) -> float:
        if len(x) == 0:
            return float("nan")
        k = min(int(k), len(x))
        return float(np.mean(x[-k:]))
    
    def add_transition(self, s: int, a: int, s2: int) -> None:
        """Record the most recent transition for logging.

        Parameters
        ----------
        s
            Current state index.
        a
            Action index.
        s2
            Next state index.

        """
        self.last_s = int(s)
        self.last_a = int(a)
        self.last_s2 = int(s2)

    def maybe_log(
        self,
        step: int,
        n_steps: int,
        eps: float,
        buf_len: int,
        lr: float,
    ) -> None:
        """Print one-line status if it's time."""
        if step % int(self.log_every) != 0:
            return
        now = time.time()
        dt = max(1e-9, now - self.last_log_t)
        self.last_log_t = now
        sps = float(self.log_every) / dt
        elapsed = now - self.t0

        r_mean = self._mean_last(self.ep_rets, self.ret_window)
        l_mean = self._mean_last_int(self.ep_lens, self.ret_window)
        loss_mean = self._mean_last(self.losses, self.loss_window)

        msg = (
            f"[{step:>7d}/{n_steps}]  "
            f"eps={eps:0.3f}  "
            f"buf={buf_len:>7d}  "
            f"R{self.ret_window}={r_mean:>7.3f}  "
            f"L{self.ret_window}={l_mean:>6.1f}  "
            f"loss{self.loss_window}={loss_mean:>8.8f}  "
            f"lr={lr:.1e}  "
            f"{sps:>7.1f} steps/s  "
            f"t={elapsed:>6.1f}s"
        )

        a_names = ("N", "E", "S", "W")
        s = int(self.last_s)
        a = int(self.last_a)
        s2 = int(self.last_s2)

        if s >= 0 and a >= 0 and s2 >= 0:
            node = s + 1
            node2 = s2 + 1
            a_str = a_names[a] if 0 <= a < len(a_names) else str(a)
            wall = int(s2 == s)
            msg += (
                f"  last: s={s:>3d}(n={node:>3d})"
                f" a={a_str:>2s}"
                f" s'={s2:>3d}(n={node2:>3d})"
                f" wall={wall}"
            )

        print(msg)