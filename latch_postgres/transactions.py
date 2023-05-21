import asyncio
import random
from functools import wraps
from typing import Callable

import psycopg.errors
from opentelemetry.trace import get_tracer

tracer = get_tracer(__name__)


class MaxBackoffsExceededError(RuntimeError):
    ...


def trans_failure_backoff(
    delay_quant: float = 0.1,
    *,
    max_delay: float | None = None,
    max_backoffs: int | None = None,
):
    def decorator(f: Callable):
        @wraps(f)
        async def res(*args, **kwargs):
            with tracer.start_as_current_span("transaction failure backoff") as s:
                fails = 0
                slot_exp = 1
                while True:
                    try:
                        s.set_attribute("fails", fails)
                        x = f(*args, **kwargs)
                        if asyncio.iscoroutine(x):
                            return await x
                        return x
                    except (
                        psycopg.errors.SerializationFailure,
                        psycopg.errors.DeadlockDetected,
                    ):
                        # https://en.wikipedia.org/wiki/Exponential_backoff
                        slot_exp *= 2
                        fails += 1
                        if max_backoffs is not None and fails >= max_backoffs:
                            raise MaxBackoffsExceededError()

                        slot = random.randint(0, slot_exp - 1)
                        delay = slot * delay_quant
                        if max_delay is not None and delay > max_delay:
                            delay = max_delay

                        s.set_attribute("backoff", delay)
                        await asyncio.sleep(delay)

        return res

    return decorator
