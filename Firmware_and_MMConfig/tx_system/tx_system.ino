/*
 * Updated firmware for Arduino Uno Rev3
 * Sensirion SLF3S-0600F (I2C) & Digital inputs
 */

#include <Arduino.h>
#include <SensirionI2cSf06Lf.h>
#include <Wire.h>

SensirionI2cSf06Lf sensor;

const int wastePin   = 2;
const int wastePin2  = 3;
const int BUTTON_PIN = 4;

static int16_t error;

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        delay(100);
    }

    pinMode(wastePin, INPUT_PULLUP);
    pinMode(wastePin2, INPUT_PULLUP);
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    // Standard I2C initialization (Do NOT force SDA/SCL LOW while Wire is running)
    Wire.begin();
    Wire.setClock(400000); // 400 kHz Fast-Mode
    
    // Pass standard I2C address for SLF3S
    sensor.begin(Wire, SLF3S_0600F_I2C_ADDR_08);

    sensor.stopContinuousMeasurement();
    delay(100);
    
    uint32_t productIdentifier = 0;
    uint8_t serialNumber[8] = {0};
    error = sensor.readProductIdentifier(productIdentifier, serialNumber, 8);
    if (error != NO_ERROR) {
        Serial.print("Error reading product identifier: ");
        Serial.println(error);
    }
    
    // Start continuous measurement for H2O
    sensor.startH2oContinuousMeasurement();
}

void loop() {
    int wasteState  = digitalRead(wastePin);
    int wasteState2 = digitalRead(wastePin2);
    int buttonState = digitalRead(BUTTON_PIN);

    float aFlow = 0.0;
    float aTemperature = 0.0;
    uint16_t aSignalingFlags = 0u;
    
    // Read using SLF3S-0600F scaling (Correct scale factor for 0600D variants)
    error = sensor.readMeasurementData(INV_FLOW_SCALE_FACTORS_SLF3S_0600F,
                                       aFlow, aTemperature, aSignalingFlags);

    if (error == NO_ERROR) {
        // Apply a zero-flow deadband threshold (ignore noise below 0.1 ul/min)
        if (abs(aFlow) < 0.10) {
            aFlow = 0.0;
        }

        Serial.print("Waste1:");     Serial.print(wasteState);                                  Serial.print("\t");
        Serial.print("Waste2:");     Serial.print(wasteState2);                                 Serial.print("\t");
        Serial.print("Button:");     Serial.print(buttonState == LOW ? "PRESSED" : "RELEASED"); Serial.print("\t");
        Serial.print("FlowRate:");   Serial.print(aFlow, 3);                                   Serial.print("\t");
        Serial.print("Temp:");       Serial.print(aTemperature, 2);                             Serial.print("\t");
        Serial.print("Flags:");      Serial.println(aSignalingFlags);
    } else {
        Serial.print("Read Error: ");
        Serial.println(error);
    }
    
    delay(50); // Reading interval matching sensor measurement cycle (~2ms native update rate)
}