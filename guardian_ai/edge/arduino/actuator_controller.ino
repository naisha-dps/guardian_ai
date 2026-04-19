/*
 * Guardian AI - Arduino Actuator Controller
 * ==========================================
 * Controls physical deterrents and reads sensors.
 * Communicates with Raspberry Pi via Serial (JSON protocol).
 *
 * Hardware:
 *   Pin 2  → PIR Sensor (HC-SR501) OUT
 *   Pin 3  → Siren / Buzzer (via relay)
 *   Pin 4  → Flash Light (via relay / MOSFET)
 *   Pin 5  → Ultrasonic Deterrent Module (40kHz)
 *   Pin 13 → Built-in LED (status)
 *   Pin A0 ← LDR (light sensor for night mode)
 *
 * Serial Protocol (9600 baud):
 *   Receive: {"cmd":"siren","state":1,"duration":10}
 *   Send:    {"status":"ok","sensor":"pir","value":1}
 *
 * Author: Guardian AI Team
 * Version: 1.0.0
 */

#include <Arduino.h>
#include <ArduinoJson.h>  // Install: Library Manager → ArduinoJson by Benoit Blanchon

// ─── Pin Definitions ────────────────────────────────────────────────────────

#define PIR_PIN         2
#define SIREN_PIN       3
#define FLASH_PIN       4
#define ULTRASONIC_PIN  5
#define STATUS_LED      13
#define LDR_PIN         A0

// ─── Timing ─────────────────────────────────────────────────────────────────

unsigned long sirenOffTime     = 0;
unsigned long flashOffTime     = 0;
unsigned long ultrasonicOffTime = 0;
unsigned long lastPirSent      = 0;
unsigned long lastStatusSent   = 0;

const unsigned long PIR_DEBOUNCE_MS    = 500;
const unsigned long STATUS_INTERVAL_MS = 5000;  // Send status every 5s

// ─── State ───────────────────────────────────────────────────────────────────

bool sirenActive      = false;
bool flashActive      = false;
bool ultrasonicActive = false;
int  lastPirState     = LOW;

// ─── Setup ───────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(9600);
  while (!Serial) {}  // Wait for serial connection

  // Configure pins
  pinMode(PIR_PIN, INPUT);
  pinMode(SIREN_PIN, OUTPUT);
  pinMode(FLASH_PIN, OUTPUT);
  pinMode(ULTRASONIC_PIN, OUTPUT);
  pinMode(STATUS_LED, OUTPUT);

  // All actuators OFF at startup
  digitalWrite(SIREN_PIN, LOW);
  digitalWrite(FLASH_PIN, LOW);
  digitalWrite(ULTRASONIC_PIN, LOW);
  digitalWrite(STATUS_LED, LOW);

  // Startup blink
  for (int i = 0; i < 3; i++) {
    digitalWrite(STATUS_LED, HIGH);
    delay(200);
    digitalWrite(STATUS_LED, LOW);
    delay(200);
  }

  sendStatus("ready", "boot", 1);
}

// ─── Main Loop ───────────────────────────────────────────────────────────────

void loop() {
  // 1. Read incoming commands from Pi
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      parseCommand(line);
    }
  }

  // 2. Auto-shutoff for timed actuators
  unsigned long now = millis();

  if (sirenActive && sirenOffTime > 0 && now >= sirenOffTime) {
    sirenOff();
  }
  if (flashActive && flashOffTime > 0 && now >= flashOffTime) {
    flashOff();
  }
  if (ultrasonicActive && ultrasonicOffTime > 0 && now >= ultrasonicOffTime) {
    ultrasonicOff();
  }

  // 3. Read PIR sensor and send to Pi
  int pirState = digitalRead(PIR_PIN);
  if (pirState != lastPirState && (now - lastPirSent) > PIR_DEBOUNCE_MS) {
    lastPirState = pirState;
    lastPirSent = now;
    sendStatus("ok", "pir", pirState);

    // Blink status LED on motion
    if (pirState == HIGH) {
      digitalWrite(STATUS_LED, HIGH);
    } else {
      digitalWrite(STATUS_LED, LOW);
    }
  }

  // 4. Periodic status heartbeat to Pi
  if (now - lastStatusSent > STATUS_INTERVAL_MS) {
    lastStatusSent = now;
    sendPeriodicStatus();
  }

  // 5. Ultrasonic pulse generation (40kHz square wave)
  if (ultrasonicActive) {
    // Toggle pin at 40kHz = 25µs half-period
    // (For real 40kHz, use hardware PWM or timer interrupt)
    digitalWrite(ULTRASONIC_PIN, HIGH);
    delayMicroseconds(12);
    digitalWrite(ULTRASONIC_PIN, LOW);
    delayMicroseconds(12);
  }
}

// ─── Command Parser ──────────────────────────────────────────────────────────

void parseCommand(String jsonStr) {
  /*
   * Parse JSON command from Raspberry Pi.
   * Format: {"cmd": "siren", "state": 1, "duration": 10}
   * 
   * Commands: siren, flash, ultrasonic
   * State: 1=ON, 0=OFF
   * Duration: seconds (0=indefinite)
   */
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, jsonStr);

  if (err) {
    sendStatus("error", "parse", 0);
    return;
  }

  const char* cmd   = doc["cmd"]      | "";
  int state         = doc["state"]    | 0;
  int duration      = doc["duration"] | 0;

  if (strcmp(cmd, "siren") == 0) {
    if (state) sirenOn(duration);
    else        sirenOff();
  }
  else if (strcmp(cmd, "flash") == 0) {
    if (state) flashOn(duration);
    else        flashOff();
  }
  else if (strcmp(cmd, "ultrasonic") == 0) {
    if (state) ultrasonicOn(duration);
    else        ultrasonicOff();
  }
  else if (strcmp(cmd, "all_off") == 0) {
    sirenOff();
    flashOff();
    ultrasonicOff();
  }
  else if (strcmp(cmd, "status") == 0) {
    sendPeriodicStatus();
  }

  sendStatus("ok", cmd, state);
}

// ─── Actuator Controls ───────────────────────────────────────────────────────

void sirenOn(int durationSec) {
  digitalWrite(SIREN_PIN, HIGH);
  sirenActive = true;
  sirenOffTime = (durationSec > 0) ? millis() + (durationSec * 1000UL) : 0;
}

void sirenOff() {
  digitalWrite(SIREN_PIN, LOW);
  sirenActive = false;
  sirenOffTime = 0;
}

void flashOn(int durationSec) {
  digitalWrite(FLASH_PIN, HIGH);
  flashActive = true;
  flashOffTime = (durationSec > 0) ? millis() + (durationSec * 1000UL) : 0;
}

void flashOff() {
  digitalWrite(FLASH_PIN, LOW);
  flashActive = false;
  flashOffTime = 0;
}

void ultrasonicOn(int durationSec) {
  // ultrasonicActive flag causes 40kHz generation in main loop
  ultrasonicActive = true;
  ultrasonicOffTime = (durationSec > 0) ? millis() + (durationSec * 1000UL) : 0;
}

void ultrasonicOff() {
  ultrasonicActive = false;
  ultrasonicOffTime = 0;
  digitalWrite(ULTRASONIC_PIN, LOW);
}

// ─── Status Reporting ────────────────────────────────────────────────────────

void sendStatus(const char* status, const char* sensor, int value) {
  StaticJsonDocument<128> doc;
  doc["status"] = status;
  doc["sensor"] = sensor;
  doc["value"]  = value;
  doc["ms"]     = millis();
  serializeJson(doc, Serial);
  Serial.println();
}

void sendPeriodicStatus() {
  // Read light level (0=dark, 1023=bright)
  int ldrValue = analogRead(LDR_PIN);
  bool isNight = (ldrValue < 200);

  StaticJsonDocument<256> doc;
  doc["status"]        = "heartbeat";
  doc["pir"]           = digitalRead(PIR_PIN);
  doc["siren"]         = sirenActive;
  doc["flash"]         = flashActive;
  doc["ultrasonic"]    = ultrasonicActive;
  doc["ldr"]           = ldrValue;
  doc["night_mode"]    = isNight;
  doc["uptime_ms"]     = millis();
  serializeJson(doc, Serial);
  Serial.println();
}

/*
 * ─── Wiring Guide ──────────────────────────────────────────────────────────
 *
 * PIR Sensor (HC-SR501):
 *   VCC → Arduino 5V
 *   GND → Arduino GND
 *   OUT → Arduino Pin 2
 *
 * Relay Module (for Siren/Flash):
 *   VCC → Arduino 5V
 *   GND → Arduino GND
 *   IN1 → Arduino Pin 3 (siren)
 *   IN2 → Arduino Pin 4 (flash)
 *   Siren: COM-NC terminals connected to 12V siren
 *   Flash: COM-NC terminals connected to LED spotlight
 *
 * Ultrasonic Deterrent:
 *   Direct 5V buzzer or ultrasonic transducer on Pin 5
 *   For more power: use transistor/MOSFET driver
 *
 * LDR (Light-Dependent Resistor):
 *   10k resistor voltage divider → A0
 *
 * Arduino to Raspberry Pi:
 *   USB cable (provides power + serial)
 *   OR Arduino TX/RX → Pi GPIO 14/15 (UART)
 */
