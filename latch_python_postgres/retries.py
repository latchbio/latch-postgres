from random import randint


class CABackoff:
    """Collision-avoidance Exponential Backoff

    Non-deterministically chooses a re-transmission slot with exponential backoff
    See: https://en.wikipedia.org/wiki/Exponential_backoff#Collision_avoidance

    The expected delay is `delay_quant / 2 * (2^retries - 1)`
    """

    def __init__(self, delay_quant: float, max_wait_time: float | None):
        self._retries = 0

        self._acc_wait_time = 0
        self._max_wait_time = max_wait_time

        self._delay_quant = delay_quant

        self._slot_exp = 1

    @property
    def retries(self):
        return self._retries

    @property
    def acc_wait_time(self):
        return self._acc_wait_time

    def retry(self) -> float | None:
        if (
            self._max_wait_time is not None
            and self._max_wait_time - self.acc_wait_time < 0.001
        ):
            return

        self._slot_exp *= 2
        slot = randint(0, self._slot_exp - 1)
        delay = slot * self._delay_quant

        if (
            self._max_wait_time is not None
            and self._acc_wait_time + delay > self._max_wait_time
        ):
            delay = self._max_wait_time - self._acc_wait_time

        self._retries += 1
        self._acc_wait_time += delay

        return delay
