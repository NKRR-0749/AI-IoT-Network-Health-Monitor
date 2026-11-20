"""Block: Python device emulator ("N virtual ESP32s").  Label: EMULATED.

Each virtual device behaves like the ESP32 firmware in firmware/esp32_telemetry:
  * publishes numbered telemetry every `interval` seconds (QoS 0 or 1)
  * sends an RTT ping every 2 s and reports the measured round trip
  * sends a UDP probe every 0.5 s (network-layer loss signal)
  * has a Last Will ("offline") and announces "online" on every connect

Why emulate? Wokwi runs ONE simulated board and cannot sit behind a fault
injector. The emulator can run 10-50 devices through the impairment proxy,
which is what the experiments need.

QoS behaviour (this is the real paho-mqtt client, not a model of it):
  * QoS 0 published while disconnected  -> dropped (that seq is lost)
  * QoS 1 published while disconnected  -> queued, sent after reconnect (late, not lost)

Run (directly to the broker):
    python -m iotmon.emulator --run-id demo --devices 3
Through the fault injector:
    python -m iotmon.emulator --run-id demo --broker-port 18830 --udp-port 19990
"""
import argparse
import json
import random
import socket
import threading
import time

import paho.mqtt.client as mqtt

from .config import settings, topic


class VirtualDevice:
    def __init__(self, run_id, device_id, *, boot_id, qos, interval, broker_host, broker_port,
                 udp_host, udp_port, keepalive, ping_interval=2.0, ping_timeout=3.0, probe_interval=0.5):
        self.run_id, self.device_id, self.boot_id, self.qos = run_id, device_id, boot_id, qos
        self.interval, self.ping_interval, self.ping_timeout = interval, ping_interval, ping_timeout
        self.probe_interval = probe_interval
        self.udp_addr = (udp_host, udp_port)
        self.t0 = time.monotonic()                     # "boot" moment for uptime_ms
        self.seq = 0
        self.probe_seq = 0
        self.udp_seq = 0
        self.pending_pings: dict[int, float] = {}
        self.lock = threading.Lock()
        self.stats = {"published": 0, "dropped_offline": 0}
        self.stop_event = threading.Event()

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{run_id}-{device_id}")
        self.client.will_set(topic(run_id, device_id, "status"), "offline", qos=1, retain=True)
        self.client.reconnect_delay_set(min_delay=1, max_delay=4)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.broker = (broker_host, broker_port, keepalive)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def uptime_ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            return
        client.publish(topic(self.run_id, self.device_id, "status"), "online", qos=1, retain=True)
        client.subscribe(topic(self.run_id, self.device_id, "pong"), qos=0)

    def _on_message(self, client, userdata, msg):
        try:
            d = json.loads(msg.payload)
            seq = int(d["probe_seq"])
        except (ValueError, KeyError, TypeError):
            return
        with self.lock:
            sent = self.pending_pings.pop(seq, None)
        if sent is not None:
            rtt = (time.monotonic() - sent) * 1000
            client.publish(topic(self.run_id, self.device_id, "rtt"),
                           json.dumps({"probe_seq": seq, "rtt_ms": round(rtt, 2)}), qos=0)

    def _publish_telemetry(self):
        payload = json.dumps({"device_id": self.device_id, "seq": self.seq, "boot_id": self.boot_id,
                              "uptime_ms": self.uptime_ms(), "qos": self.qos,
                              "interval_ms": int(self.interval * 1000),
                              "rssi": -60 + random.randint(-3, 3)})    # SYNTHETIC rssi
        info = self.client.publish(topic(self.run_id, self.device_id, "telemetry"), payload, qos=self.qos)
        if info.rc == mqtt.MQTT_ERR_NO_CONN and self.qos == 0:
            self.stats["dropped_offline"] += 1
        else:
            self.stats["published"] += 1
        self.seq += 1                                  # seq advances even if the message was dropped

    def _ping(self):
        now = time.monotonic()
        with self.lock:
            expired = [s for s, t in self.pending_pings.items() if now - t > self.ping_timeout]
            for s in expired:
                del self.pending_pings[s]
            self.pending_pings[self.probe_seq] = now
        for s in expired:                              # report lost pings as rtt = null
            self.client.publish(topic(self.run_id, self.device_id, "rtt"),
                                json.dumps({"probe_seq": s, "rtt_ms": None}), qos=0)
        self.client.publish(topic(self.run_id, self.device_id, "ping"),
                            json.dumps({"probe_seq": self.probe_seq, "t": self.uptime_ms()}), qos=0)
        self.probe_seq += 1

    def _udp_probe(self):
        pkt = json.dumps({"run_id": self.run_id, "device_id": self.device_id, "boot_id": self.boot_id,
                          "probe_seq": self.udp_seq, "uptime_ms": self.uptime_ms()}).encode()
        try:
            self.udp.sendto(pkt, self.udp_addr)
        except OSError:
            pass
        self.udp_seq += 1

    def run(self, duration: float | None):
        host, port, keepalive = self.broker
        self.client.connect_async(host, port, keepalive=keepalive)
        self.client.loop_start()
        start = time.monotonic()
        next_tel = next_ping = next_probe = start + random.uniform(0, self.interval)  # spread devices out
        while not self.stop_event.is_set():
            now = time.monotonic()
            if duration is not None and now - start >= duration:
                break
            if now >= next_tel:
                self._publish_telemetry()
                next_tel += self.interval
            if now >= next_ping:
                self._ping()
                next_ping += self.ping_interval
            if now >= next_probe:
                self._udp_probe()
                next_probe += self.probe_interval
            time.sleep(max(0.0, min(next_tel, next_ping, next_probe) - time.monotonic()))
        time.sleep(1.0)                                # let queued QoS 1 messages drain
        self.client.loop_stop()
        self.client.disconnect()


def main():
    ap = argparse.ArgumentParser(description="Emulate N MQTT IoT devices")
    ap.add_argument("--run-id", default="demo")
    ap.add_argument("--devices", type=int, default=3)
    ap.add_argument("--prefix", default="dev")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between telemetry messages")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1])
    ap.add_argument("--broker-host", default=settings.mqtt_host)
    ap.add_argument("--broker-port", type=int, default=settings.mqtt_port)
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=settings.udp_probe_port)
    ap.add_argument("--keepalive", type=int, default=5)
    ap.add_argument("--boot-id", type=int, default=int(time.time()) % 100000)
    ap.add_argument("--duration", type=float, default=None, help="seconds; default runs until Ctrl+C")
    args = ap.parse_args()

    devices = [VirtualDevice(args.run_id, f"{args.prefix}-{i:02d}", boot_id=args.boot_id, qos=args.qos,
                             interval=args.interval, broker_host=args.broker_host, broker_port=args.broker_port,
                             udp_host=args.udp_host, udp_port=args.udp_port, keepalive=args.keepalive)
               for i in range(1, args.devices + 1)]
    threads = [threading.Thread(target=d.run, args=(args.duration,), daemon=True) for d in devices]
    for t in threads:
        t.start()
    print(f"[emulator] {len(devices)} device(s), run={args.run_id}, qos={args.qos}, interval={args.interval}s")
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        for d in devices:
            d.stop_event.set()
        for t in threads:
            t.join(timeout=5)
    total = sum(d.stats["published"] for d in devices)
    dropped = sum(d.stats["dropped_offline"] for d in devices)
    print(f"[emulator] done: published={total} dropped_while_offline={dropped}")


if __name__ == "__main__":
    main()
