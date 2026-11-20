"""Block: userspace impairment proxy.   Label: EMULATED (userspace model).

Devices connect to the proxy instead of the broker. The proxy forwards bytes
both ways and degrades the link on purpose:

  TCP (MQTT) path
    delay / jitter   each chunk of bytes is held for delay +- jitter ms
    loss             a "lost segment" is modelled the way TCP experiences it:
                     the data is NOT dropped (TCP would resend it) but it is
                     held for a retransmission timeout (RTO, min 200 ms, doubling
                     on every repeated loss), and everything behind it waits too
                     (head-of-line blocking). Bytes are never reordered.
    outage 'stall'   link silently stops; data waits until the link returns
                     (a short outage: TCP survives it, MQTT keepalive may not)
    outage 'drop'    all connections are closed and new ones refused
                     (link down long enough to break TCP)
  UDP (probe) path
    loss / delay / jitter / outage applied per datagram; lost = really gone,
    because UDP has no retransmission. This is the network-layer loss signal.

IMPORTANT for the paper: the TCP-loss behaviour here is a MODEL of TCP's
reaction to loss, not real IP loss. Real IP-level loss needs netem
(iotmon.impair.netem) on Linux/WSL2. Results must say which backend was used.
"""
import asyncio
import random
import time

from .models import make_loss_model

MIN_RTO_MS = 200.0
MAX_RETRIES = 6


class ImpairmentProxy:
    def __init__(self, tcp_listen: int, tcp_upstream: tuple, udp_listen: int | None = None,
                 udp_upstream: tuple | None = None, seed: int = 0, host: str = "127.0.0.1"):
        self.host = host
        self.tcp_listen, self.tcp_upstream = tcp_listen, tcp_upstream
        self.udp_listen, self.udp_upstream = udp_listen, udp_upstream
        self.rng = random.Random(seed)
        self.params = {"loss_pct": 0.0, "burst_len": 0, "delay_ms": 0.0, "jitter_ms": 0.0, "outage": None}
        self.link_up = asyncio.Event()
        self.link_up.set()
        self._conns: set = set()
        self._udp_models: dict = {}
        self.stats = {"tcp_conns": 0, "tcp_refused": 0, "tcp_loss_events": 0, "udp_in": 0, "udp_dropped": 0}
        self._server = None
        self._udp_transport = None

    # ---------------- control ----------------
    def set_params(self, **params):
        """Called by the schedule runner at every phase change."""
        new = {**{"loss_pct": 0.0, "burst_len": 0, "delay_ms": 0.0, "jitter_ms": 0.0, "outage": None}, **params}
        loss_changed = (new["loss_pct"], new["burst_len"]) != (self.params["loss_pct"], self.params["burst_len"])
        self.params = new
        if loss_changed:                       # fresh loss models with the new settings
            for c in list(self._conns):
                c.reset_models()
            self._udp_models.clear()
        if new["outage"] is None:
            self.link_up.set()
        else:
            self.link_up.clear()
            if new["outage"] == "drop":
                for c in list(self._conns):
                    c.close()

    def _delay_s(self) -> float:
        p = self.params
        d = p["delay_ms"] + (self.rng.uniform(-p["jitter_ms"], p["jitter_ms"]) if p["jitter_ms"] else 0.0)
        return max(0.0, d) / 1000.0

    def _new_model(self):
        return make_loss_model(self.params["loss_pct"], self.params["burst_len"], random.Random(self.rng.random()))

    # ---------------- TCP ----------------
    async def _handle(self, reader, writer):
        if self.params["outage"] == "drop":
            self.stats["tcp_refused"] += 1
            writer.close()
            return
        try:
            up_reader, up_writer = await asyncio.open_connection(*self.tcp_upstream)
        except OSError:
            writer.close()
            return
        conn = _Connection(self, reader, writer, up_reader, up_writer)
        self._conns.add(conn)
        self.stats["tcp_conns"] += 1
        try:
            await conn.run()
        finally:
            self._conns.discard(conn)

    # ---------------- UDP ----------------
    def _udp_received(self, data: bytes, addr):
        self.stats["udp_in"] += 1
        model = self._udp_models.get(addr)
        if model is None:
            model = self._udp_models[addr] = self._new_model()
        if self.params["outage"] is not None or model.drop():
            self.stats["udp_dropped"] += 1
            return
        loop = asyncio.get_running_loop()
        loop.call_later(self._delay_s(), self._udp_transport.sendto, data, self.udp_upstream)

    async def start(self):
        self._server = await asyncio.start_server(self._handle, self.host, self.tcp_listen)
        if self.udp_listen:
            loop = asyncio.get_running_loop()
            proxy = self

            class _Udp(asyncio.DatagramProtocol):
                def datagram_received(self, data, addr):
                    proxy._udp_received(data, addr)

            self._udp_transport, _ = await loop.create_datagram_endpoint(_Udp, local_addr=(self.host, self.udp_listen))

    async def stop(self):
        for c in list(self._conns):
            c.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._udp_transport:
            self._udp_transport.close()


class _Connection:
    """One device connection = two one-way pipes (device->broker, broker->device)."""

    def __init__(self, proxy, reader, writer, up_reader, up_writer):
        self.proxy = proxy
        self.pipes = [_Pipe(proxy, reader, up_writer), _Pipe(proxy, up_reader, writer)]
        self.writers = [writer, up_writer]

    def reset_models(self):
        for p in self.pipes:
            p.model = self.proxy._new_model()

    def close(self):
        for w in self.writers:
            try:
                w.close()
            except Exception:
                pass

    async def run(self):
        tasks = [asyncio.create_task(t) for p in self.pipes for t in (p.read_loop(), p.write_loop())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        self.close()
        for t in tasks:
            t.cancel()


class _Pipe:
    def __init__(self, proxy, reader, writer):
        self.proxy, self.reader, self.writer = proxy, reader, writer
        self.queue: asyncio.Queue = asyncio.Queue()
        self.model = proxy._new_model()
        self.last_release = 0.0

    async def read_loop(self):
        while True:
            data = await self.reader.read(4096)
            if not data:
                await self.queue.put(None)
                return
            now = time.monotonic()
            hold = self.proxy._delay_s()
            # TCP view of loss: each loss of this segment costs one RTO (doubling), data still arrives in order
            rto = max(MIN_RTO_MS, 2 * self.proxy.params["delay_ms"]) / 1000.0
            tries = 0
            while self.model.drop() and tries < MAX_RETRIES:
                hold += rto
                rto *= 2
                tries += 1
            if tries:
                self.proxy.stats["tcp_loss_events"] += 1
            release = max(self.last_release, now + hold)   # never overtake earlier bytes
            self.last_release = release
            await self.queue.put((release, data))

    async def write_loop(self):
        while True:
            item = await self.queue.get()
            if item is None:
                self.writer.close()
                return
            release, data = item
            wait = release - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            await self.proxy.link_up.wait()                # 'stall' outage: hold here
            self.writer.write(data)
            await self.writer.drain()
