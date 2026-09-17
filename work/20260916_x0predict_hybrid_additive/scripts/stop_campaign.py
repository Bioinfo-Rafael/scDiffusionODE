"""Stop only the named Linux campaign's launcher/workers and descendants.

No checkpoint creation is implied: resume uses the latest complete saved bundle.
This standalone script needs only Python's standard library, not the ML env.
"""
import argparse
import os
from pathlib import Path
import signal
import time

PACKAGE = "work.20260916_x0predict_hybrid_additive"


def identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return int(fields[1]), fields[19], fields[0]  # parent, start time, state
    except (OSError, ValueError, IndexError):
        return None


def snapshot():
    result = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            argv = path.joinpath("cmdline").read_bytes().decode().split("\0")
            info = identity(int(path.name))
            if info:
                result[int(path.name)] = (argv, info)
        except (OSError, UnicodeError):
            pass
    return result


def select(processes, campaign):
    selected = {pid for pid, (argv, _) in processes.items()
                if any(a in (PACKAGE + ".scripts.run_all", PACKAGE + ".cli") for a in argv)
                and any(a == campaign or f"/{campaign}/" in a for a in argv)}
    for pid in list(selected):
        parent = processes[pid][1][0]
        if parent in processes and PACKAGE + ".scripts.run_all" in processes[parent][0]:
            selected.add(parent)
    return selected


def descendants(processes, selected):
    selected = set(selected)
    while True:
        children = {p for p, (_, info) in processes.items() if info[0] in selected}
        if children <= selected:
            return selected
        selected |= children


def alive(pid, processes):
    now = identity(pid)
    return bool(now and now[1] == processes[pid][1][1] and now[2] != "Z")


def send(pid, sig, processes):
    if alive(pid, processes):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def stop(campaign, *, dry_run=False, timeout=15):
    if not Path("/proc").is_dir():
        raise RuntimeError("run this command on the Linux training host")
    processes = snapshot()
    selected = select(processes, campaign)
    launchers = {p for p in selected if PACKAGE + ".scripts.run_all" in processes[p][0]}
    if dry_run:
        selected = descendants(processes, selected)
        for pid in sorted(selected):
            print(pid, " ".join(processes[pid][0]))
        return 0
    # Freeze launchers first so a failed worker cannot trigger the next condition.
    try:
        for pid in launchers:
            send(pid, signal.SIGSTOP, processes)
        refreshed = snapshot()
        for pid, item in refreshed.items():
            if pid not in selected:
                processes[pid] = item
        selected = descendants(processes, selected)
        for pid in sorted(selected - launchers):
            send(pid, signal.SIGTERM, processes)
        for pid in launchers:
            send(pid, signal.SIGTERM, processes)
    finally:
        for pid in launchers:
            send(pid, signal.SIGCONT, processes)
    deadline = time.monotonic() + timeout
    remaining = sorted(p for p in selected if alive(p, processes))
    while remaining and time.monotonic() < deadline:
        time.sleep(.2)
        remaining = sorted(p for p in selected if alive(p, processes))
    print("Target PIDs:", sorted(selected))
    print("Still running:", remaining)
    print("Checkpoint/result files preserved; only saved complete bundles can be resumed.")
    return int(bool(remaining))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if Path(args.campaign).name != args.campaign or not args.campaign.startswith("additive_"):
        p.error("provide the exact additive_... campaign name")
    return stop(args.campaign, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
