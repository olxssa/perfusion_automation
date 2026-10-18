#include <Servo.h>
#include <Adafruit_NeoPixel.h>
#include <SoftwareSerial.h>

// ==========================================
// PIN DEFINITIONS & HARDWARE STRUCTS
// ==========================================
// Core Peripherals
const int neoPixelPin = 6;
const int servoPin = 10;
const int microServoPin = 11;
const int RELAY_PIN = 2; // D2 mapped as your Olimex Socket Control Pin

// Valve Control Pins
const int VALVE1_PIN = 12; // Dedicated pin for Valve 1
const int VALVE2_PIN = A0; // Dedicated pin for Valve 2 (Analog pin used as digital IO)

// Pump 1 Pins
#define DIR1_PIN  3
#define STEP1_PIN 4
#define EN1_PIN   5

// Pump 2 Pins
#define DIR2_PIN  7
#define STEP2_PIN 8
#define EN2_PIN   9

// NeoPixel Configuration
const int numPixels = 24;
const int frameDelay = 20; 

Servo myServo;
Servo microServo;
Adafruit_NeoPixel ring(numPixels, neoPixelPin, NEO_GRB + NEO_KHZ800);

bool rainbowMode = false;
uint16_t firstPixelHue = 0;
unsigned long lastUpdate = 0;

// Pump Parameters & States
const int STEPS_PER_REV = 1600; 
unsigned long stepDelayUS = 16000; 

bool pump1Running = false;
bool pump2Running = false;

// Function Prototypes
void moveServo(Servo &s, int pin, int angle, String label);
void handlePumpCommand(int pump, char action, String args);
void pulseStep(int stepPin);
void dispenseSteps(int pump, int stepPin, int dirPin, int enPin, long steps);

// ==========================================
// SETUP & LOOP
// ==========================================
void setup() {
  Serial.begin(9600);

  // Initialize Socket Pin
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW); // Start off by default

  // Initialize Pump 1 Pins
  pinMode(EN1_PIN, OUTPUT);
  pinMode(STEP1_PIN, OUTPUT);
  pinMode(DIR1_PIN, OUTPUT);
  digitalWrite(EN1_PIN, HIGH); // Disabled by default
  digitalWrite(DIR1_PIN, LOW);

  // Initialize Pump 2 Pins
  pinMode(EN2_PIN, OUTPUT);
  pinMode(STEP2_PIN, OUTPUT);
  pinMode(DIR2_PIN, OUTPUT);
  digitalWrite(EN2_PIN, HIGH); // Disabled by default
  digitalWrite(DIR2_PIN, LOW);

  // Initialize Valve Pins (Off by default)
  pinMode(VALVE1_PIN, OUTPUT);
  digitalWrite(VALVE1_PIN, LOW);

  pinMode(VALVE2_PIN, OUTPUT);
  digitalWrite(VALVE2_PIN, LOW);

  ring.begin();
  ring.setBrightness(100);
  ring.show();

  Serial.println("READY");
}

void loop() {
  // --- Handle Incoming Commands ---
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // Socket Command Controls
    if (cmd == "SOCKET_ON") {
      digitalWrite(RELAY_PIN, HIGH);
      Serial.println("OK: SOCKET ON");
    }
    else if (cmd == "SOCKET_OFF") {
      digitalWrite(RELAY_PIN, LOW);
      Serial.println("OK: SOCKET OFF");
    }
    // Rainbow Controls
    else if (cmd == "RAINBOW") {
      rainbowMode = true;
      Serial.println("OK: RAINBOW START");
    }
    else if (cmd.startsWith("C")) {
    rainbowMode = false; 
    int r, g, b;
    if (sscanf(cmd.c_str(), "C %d %d %d", &r, &g, &b) == 3) {
      // Constrain RGB values strictly to valid byte ranges
      r = constrain(r, 0, 255);
      g = constrain(g, 0, 255);
      b = constrain(b, 0, 255);
      for (int i = 0; i < numPixels; i++) {
        ring.setPixelColor(i, ring.Color(r, g, b));
      }
      delayMicroseconds(50);
      ring.show();
      // If powering off, explicitly pull data pin LOW to stop floating line noise
      if (r == 0 && g == 0 && b == 0) {
        delay(1); 
        digitalWrite(neoPixelPin, LOW);
      }
      Serial.println("OK: COLOR SET");
      }
    }   
    // Brightness Handler
    else if (cmd.startsWith("B ")) {
      int percentage = cmd.substring(2).toInt();
      percentage = constrain(percentage, 0, 100); 
      
      int rawBrightness = map(percentage, 0, 100, 0, 255);
      ring.setBrightness(rawBrightness);
      ring.show(); 
      
      Serial.print("OK: BRIGHTNESS SET TO ");
      Serial.print(percentage);
      Serial.println(" %");
    }
    // Servo Controls
    else if (cmd.startsWith("A")) {
      moveServo(myServo, servoPin, cmd.substring(2).toInt(), "MAIN");
    }
    else if (cmd.startsWith("M")) {
      moveServo(microServo, microServoPin, cmd.substring(2).toInt(), "MICRO");
    }
    // Speed command handler
    else if (cmd.startsWith("DELAY ")) {
      unsigned long newDelay = cmd.substring(6).toInt();
      if (newDelay > 0) {
        stepDelayUS = newDelay;
        Serial.print("OK: STEP DELAY SET TO ");
        Serial.print(stepDelayUS);
        Serial.println(" US");
      } else {
        Serial.println("ERROR: Invalid delay value");
      }
    }
    // Pump Controls (Commands starting with '1' or '2')
    else if (cmd.length() >= 2 && (cmd.charAt(0) == '1' || cmd.charAt(0) == '2')) {
      int pumpNum = cmd.charAt(0) - '0';
      char action = toupper(cmd.charAt(1));
      handlePumpCommand(pumpNum, action, cmd.substring(2));
    }
    // --- Valve Controls (Pins 12 & A0) ---
    else if (cmd == "VALVE1_ON") {
      digitalWrite(VALVE1_PIN, HIGH);
      Serial.println("OK: VALVE 1 ON");
    }
    else if (cmd == "VALVE1_OFF") {
      digitalWrite(VALVE1_PIN, LOW);
      Serial.println("OK: VALVE 1 OFF");
    }
    else if (cmd == "VALVE2_ON") {
      digitalWrite(VALVE2_PIN, HIGH);
      Serial.println("OK: VALVE 2 ON");
    }
    else if (cmd == "VALVE2_OFF") {
      digitalWrite(VALVE2_PIN, LOW);
      Serial.println("OK: VALVE 2 OFF");
    }
    else if (cmd == "VALVES_ALL_ON") {
      digitalWrite(VALVE1_PIN, HIGH);
      digitalWrite(VALVE2_PIN, HIGH);
      Serial.println("OK: ALL VALVES ON");
    }
    else if (cmd == "VALVES_ALL_OFF") {
      digitalWrite(VALVE1_PIN, LOW);
      digitalWrite(VALVE2_PIN, LOW);
      Serial.println("OK: ALL VALVES OFF");
    }
    else {
      Serial.print("ERROR: Unknown command received: ");
      Serial.println(cmd);
    }
  }

  // --- Non-blocking Rainbow Animation ---
  if (rainbowMode) {
    if (millis() - lastUpdate > frameDelay) {
      lastUpdate = millis();
      for (int i = 0; i < ring.numPixels(); i++) {
        int pixelHue = firstPixelHue + (i * 65536L / ring.numPixels());
        ring.setPixelColor(i, ring.gamma32(ring.ColorHSV(pixelHue)));
      }
      ring.show();
      firstPixelHue += 256; 
    }
  }

  // --- Continuous Pump Operation Handlers ---
  if (pump1Running) {
    pulseStep(STEP1_PIN);
  }
  if (pump2Running) {
    pulseStep(STEP2_PIN);
  }
}

// ==========================================
// SERVO CONTROL
// ==========================================
void moveServo(Servo &s, int pin, int angle, String label) {
  if (angle >= 0 && angle <= 180) {
    s.attach(pin);
    s.write(angle);
    delay(600); 
    s.detach();
    Serial.print("OK: "); Serial.print(label); Serial.println(" MOVED");
  }
}

// ==========================================
// PUMP COMMAND PROCESSING
// ==========================================
void handlePumpCommand(int pump, char action, String args) {
  int enPin   = (pump == 1) ? EN1_PIN   : EN2_PIN;
  int dirPin  = (pump == 1) ? DIR1_PIN  : DIR2_PIN;
  int stepPin = (pump == 1) ? STEP1_PIN : STEP2_PIN;
  bool &runningState = (pump == 1) ? pump1Running : pump2Running;

  if (action == 'F') {
    digitalWrite(dirPin, LOW);   // Forward
    digitalWrite(enPin, LOW);    // Enable motor
    runningState = true;
    Serial.print(F("OK: Pump ")); Serial.print(pump); Serial.println(F(" Forward"));
  } 
  else if (action == 'R') {
    digitalWrite(dirPin, HIGH);  // Reverse
    digitalWrite(enPin, LOW);    // Enable motor
    runningState = true;
    Serial.print(F("OK: Pump ")); Serial.print(pump); Serial.println(F(" Reverse"));
  } 
  else if (action == 'S') {
    runningState = false;
    digitalWrite(enPin, HIGH);   // Disable driver
    Serial.print(F("OK: Pump ")); Serial.print(pump); Serial.println(F(" Stopped"));
  } 
  else if (action == 'D') {
    runningState = false;        // Pause continuous mode if active
    long steps = args.toInt();   // Supports positive and negative values
    if (steps != 0) {
      dispenseSteps(pump, stepPin, dirPin, enPin, steps);
    }
  }
}

// ==========================================
// PUMP MOTION FUNCTIONS
// ==========================================
void pulseStep(int stepPin) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(stepDelayUS / 2);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(stepDelayUS / 2);
}

void dispenseSteps(int pump, int stepPin, int dirPin, int enPin, long steps) {
  bool isReverse = steps < 0;
  long totalSteps = abs(steps);

  digitalWrite(dirPin, isReverse ? HIGH : LOW);
  digitalWrite(enPin, LOW); // Enable driver

  Serial.print(F("Status: Pump "));
  Serial.print(pump);
  Serial.print(F(" dispensing "));
  Serial.print(totalSteps);
  Serial.println(isReverse ? F(" steps REVERSE...") : F(" steps FORWARD..."));

  for (long i = 0; i < totalSteps; i++) {
    pulseStep(stepPin);
  }

  digitalWrite(enPin, HIGH); // Disable driver when complete
  Serial.print(F("OK: Pump "));
  Serial.print(pump);
  Serial.println(F(" Dispensing Complete"));
}