"""Block: netem backend (REAL IP-level impairment, Linux or WSL2 only).  Label: EMULATED (kernel netem).

netem is the Linux kernel's network emulator. Unlike the userspace proxy it
drops and delays real IP packets, so TCP's actual retransmission behaviour
is what you measure. Use it for the experiments you report in the paper.

Setup (once): run Mosquitto with a SECOND listener on port 1884 (see
experiments/mosquitto.conf). Devices connect to 1884; the collector keeps
using 1883. Only traffic on port 1884 and UDP 9999 is impaired, so the
collector's own link stays clean.

Needs root:  sudo -E python -m iotmon.experiment run --backend netem ...

STATUS: written to match netem's documented syntax, but NOT run in the build
environment (its kernel has no netem). Test it with a small run first.
"""
import shlex
import subprocess

from .models import GilbertElliottLoss

DEV = "lo"
IMPAIRED_TCP_PORT = 1884
IMPAIRED_UDP_PORT = 9999


def _run(cmd: str, check=True):
    return subprocess.run(shlex.split(cmd), check=check, capture_output=True, text=True)


def netem_args(params: dict) -> str:
    """Translate one phase's params into netem arguments."""
    if params.get("outage"):
        return "loss 100%"                       # both outage kinds become a full blackout at IP level
    parts = []
    delay, jitter = params.get("delay_ms", 0), params.get("jitter_ms", 0)
    if delay:
        parts.append(f"delay {delay}ms" + (f" {jitter}ms distribution normal" if jitter else ""))
    loss = params.get("loss_pct", 0)
    if loss:
        burst = params.get("burst_len", 0)
        parts.append(GilbertElliottLoss(loss / 100, burst).netem_args() if burst and burst > 1 else f"loss {loss}%")
    return " ".join(parts) or "delay 0ms"


def setup():
    teardown()
    _run(f"tc qdisc add dev {DEV} root handle 1: prio bands 4")
    _run(f"tc qdisc add dev {DEV} parent 1:4 handle 40: netem delay 0ms")
    for match in (f"ip dport {IMPAIRED_TCP_PORT} 0xffff", f"ip sport {IMPAIRED_TCP_PORT} 0xffff",
                  f"ip dport {IMPAIRED_UDP_PORT} 0xffff"):
        _run(f"tc filter add dev {DEV} parent 1: protocol ip prio 1 u32 match {match} flowid 1:4")


def apply(params: dict):
    _run(f"tc qdisc change dev {DEV} parent 1:4 handle 40: netem {netem_args(params)}")


def teardown():
    _run(f"tc qdisc del dev {DEV} root", check=False)
