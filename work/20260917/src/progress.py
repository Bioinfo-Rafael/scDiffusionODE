"""Timestamped phase boundaries and a heartbeat for otherwise quiet operations."""
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Event, Thread
import time


def report(message):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


@contextmanager
def phase(label, interval=30):
    started = time.monotonic()
    stopped = Event()
    report(f"START {label}")

    def heartbeat():
        while not stopped.wait(interval):
            report(f"RUNNING {label} elapsed={time.monotonic()-started:.0f}s")

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException as exc:
        report(f"FAILED {label} elapsed={time.monotonic()-started:.1f}s {type(exc).__name__}: {exc}")
        raise
    else:
        report(f"DONE {label} elapsed={time.monotonic()-started:.1f}s")
    finally:
        stopped.set()
        thread.join()
