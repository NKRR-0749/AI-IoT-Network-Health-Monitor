/*
 * AI-IoT-Network-Health-Monitor : ESP32 telemetry firmware
 * Label: SIMULATED when run in Wokwi, MEASURED when run on a real board.
 *
 * What it does (same protocol as the Python emulator, see src/iotmon/protocol.py):
 *   - connects to Wi-Fi and to the MQTT broker
 *   - publishes numbered telemetry every TELEMETRY_INTERVAL_MS
 *       topic: iot/<RUN_ID>/<DEVICE_ID>/telemetry
 *       json : {"device_id":..,"seq":..,"boot_id":..,"uptime_ms":..,"qos":0,"interval_ms":..,"rssi":..}
 *   - sends an RTT ping every 2 s, the collector echoes it, we time the round trip
 *   - announces "online" (retained) and sets a Last Will "offline"
 *   - (optional) sends UDP probe packets for network-layer loss
 *
 * Libraries: PubSubClient (by Nick O'Leary). Built-in: WiFi, Preferences, WiFiUdp.
 * Note: PubSubClient can only PUBLISH with QoS 0, so this firmware always uses QoS 0.
 */
#include <WiFi.h>
#include <WiFiUdp.h>
#include <PubSubClient.h>
#include <Preferences.h>

// ---------------- settings: change these ----------------
const char* WIFI_SSID     = "Wokwi-GUEST";     // Wokwi's simulated access point
const char* WIFI_PASSWORD = "";
// Broker: "host.wokwi.internal" reaches your laptop through the Wokwi private gateway (Wokwi for VS Code).
// Without the gateway use a public test broker such as "test.mosquitto.org" (data is public!).
const char* MQTT_HOST     = "host.wokwi.internal";
const uint16_t MQTT_PORT  = 1883;
const char* RUN_ID        = "live";
const char* DEVICE_ID     = "esp32-01";
const uint32_t TELEMETRY_INTERVAL_MS = 1000;
const uint32_t PING_INTERVAL_MS      = 2000;
const uint32_t PING_TIMEOUT_MS       = 3000;
// UDP probes: only useful when the collector's UDP port is reachable from the board.
const bool     UDP_PROBES_ENABLED    = false;
const char*    UDP_HOST              = "host.wokwi.internal";
const uint16_t UDP_PORT              = 9999;
const uint32_t UDP_PROBE_INTERVAL_MS = 500;
// --------------------------------------------------------

WiFiClient   net;
PubSubClient mqtt(net);
WiFiUDP      udp;
Preferences  prefs;

uint32_t bootId   = 0;   // increases on every reboot (stored in flash)
uint32_t seqNo    = 0;   // telemetry sequence number since boot
uint32_t pingSeq  = 0;
uint32_t udpSeq   = 0;
int32_t  pendingPing = -1;        // probe_seq we are waiting for (-1 = none)
uint32_t pendingPingSentAt = 0;
uint32_t lastTelemetry = 0, lastPing = 0, lastProbe = 0;

char tTelemetry[96], tPing[96], tPong[96], tRtt[96], tStatus[96];

void buildTopics() {
  snprintf(tTelemetry, sizeof tTelemetry, "iot/%s/%s/telemetry", RUN_ID, DEVICE_ID);
  snprintf(tPing,      sizeof tPing,      "iot/%s/%s/ping",      RUN_ID, DEVICE_ID);
  snprintf(tPong,      sizeof tPong,      "iot/%s/%s/pong",      RUN_ID, DEVICE_ID);
  snprintf(tRtt,       sizeof tRtt,       "iot/%s/%s/rtt",       RUN_ID, DEVICE_ID);
  snprintf(tStatus,    sizeof tStatus,    "iot/%s/%s/status",    RUN_ID, DEVICE_ID);
}

// Called when the collector echoes our ping back on the pong topic.
void onMessage(char* topic, byte* payload, unsigned int len) {
  char buf[128];
  unsigned int n = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
  memcpy(buf, payload, n);
  buf[n] = '\0';
  char* p = strstr(buf, "\"probe_seq\":");
  if (!p) return;
  long seq = atol(p + 12);
  if (seq == pendingPing) {
    uint32_t rtt = millis() - pendingPingSentAt;
    char out[64];
    snprintf(out, sizeof out, "{\"probe_seq\":%ld,\"rtt_ms\":%lu}", seq, (unsigned long)rtt);
    mqtt.publish(tRtt, out);
    pendingPing = -1;
  }
}

void connectWiFi() {
  Serial.print("Wi-Fi: connecting");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD, 6);   // channel 6 speeds up Wokwi
  while (WiFi.status() != WL_CONNECTED) { delay(250); Serial.print("."); }
  Serial.print(" connected, IP ");
  Serial.println(WiFi.localIP());
}

void connectMqtt() {
  while (!mqtt.connected()) {
    Serial.print("MQTT: connecting ... ");
    String clientId = String(RUN_ID) + "-" + DEVICE_ID;
    // Last Will: the broker publishes "offline" if we vanish without a clean disconnect
    if (mqtt.connect(clientId.c_str(), tStatus, 1, true, "offline")) {
      Serial.println("ok");
      mqtt.publish(tStatus, "online", true);
      mqtt.subscribe(tPong);
    } else {
      Serial.printf("failed rc=%d, retry in 2 s\n", mqtt.state());
      delay(2000);
    }
  }
}

void publishTelemetry() {
  char json[200];
  snprintf(json, sizeof json,
           "{\"device_id\":\"%s\",\"seq\":%lu,\"boot_id\":%lu,\"uptime_ms\":%lu,\"qos\":0,"
           "\"interval_ms\":%lu,\"rssi\":%d}",
           DEVICE_ID, (unsigned long)seqNo, (unsigned long)bootId, (unsigned long)millis(),
           (unsigned long)TELEMETRY_INTERVAL_MS, (int)WiFi.RSSI());
  bool ok = mqtt.publish(tTelemetry, json);   // false = not sent (e.g. disconnected) -> that seq is lost
  Serial.printf("seq=%lu %s\n", (unsigned long)seqNo, ok ? "sent" : "NOT SENT");
  seqNo++;                                     // advance even when the send failed
}

void sendPing() {
  if (pendingPing >= 0 && millis() - pendingPingSentAt > PING_TIMEOUT_MS) {
    char out[64];   // the previous ping never came back: report it as lost
    snprintf(out, sizeof out, "{\"probe_seq\":%ld,\"rtt_ms\":null}", (long)pendingPing);
    mqtt.publish(tRtt, out);
    pendingPing = -1;
  }
  if (pendingPing >= 0) return;                // one ping in flight at a time
  char out[64];
  snprintf(out, sizeof out, "{\"probe_seq\":%lu,\"t\":%lu}", (unsigned long)pingSeq, (unsigned long)millis());
  pendingPing = pingSeq++;
  pendingPingSentAt = millis();
  mqtt.publish(tPing, out);
}

void sendUdpProbe() {
  char out[160];
  snprintf(out, sizeof out,
           "{\"run_id\":\"%s\",\"device_id\":\"%s\",\"boot_id\":%lu,\"probe_seq\":%lu,\"uptime_ms\":%lu}",
           RUN_ID, DEVICE_ID, (unsigned long)bootId, (unsigned long)udpSeq++, (unsigned long)millis());
  udp.beginPacket(UDP_HOST, UDP_PORT);
  udp.write((const uint8_t*)out, strlen(out));
  udp.endPacket();
}

void setup() {
  Serial.begin(115200);
  prefs.begin("iotmon", false);
  bootId = prefs.getUInt("boot_id", 0) + 1;    // read, increment, save
  prefs.putUInt("boot_id", bootId);
  Serial.printf("\nboot_id=%lu\n", (unsigned long)bootId);

  buildTopics();
  connectWiFi();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setKeepAlive(5);
  mqtt.setCallback(onMessage);
  connectMqtt();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqtt.connected()) connectMqtt();
  mqtt.loop();

  uint32_t now = millis();
  if (now - lastTelemetry >= TELEMETRY_INTERVAL_MS) { lastTelemetry = now; publishTelemetry(); }
  if (now - lastPing >= PING_INTERVAL_MS)           { lastPing = now;      sendPing(); }
  if (UDP_PROBES_ENABLED && now - lastProbe >= UDP_PROBE_INTERVAL_MS) { lastProbe = now; sendUdpProbe(); }
}
