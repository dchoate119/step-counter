// step-counter: MPU-6050 on the ankle (header SDA/SCL, address 0x68).
//
// loop() samples at 50 Hz into a ring buffer. Python calls read_batch() to
// collect everything since its last call, one "t_ms,ax,ay,az,gx,gy,gz" line
// per sample, in g and deg/s (ranges: ±16 g, ±2000 deg/s, wide enough for running).
// imu_check() and imu_read() remain for the hardware test.

#include <Wire.h>
#include "Arduino_RouterBridge.h"

const uint8_t ADDR = 0x68;
const uint8_t REG_GYRO_CONFIG = 0x1B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_ACCEL_XOUT_H = 0x3B;
const uint8_t REG_PWR_MGMT_1 = 0x6B;
const uint8_t REG_WHO_AM_I = 0x75;

const uint8_t RANGE_MAX = 0x18;     // ±16 g / ±2000 deg/s in the config registers
const float ACCEL_LSB = 2048.0;     // counts per g at ±16 g
const float GYRO_LSB = 16.4;        // counts per deg/s at ±2000 deg/s

const unsigned long SAMPLE_MS = 20; // 50 Hz
const int BUF_SIZE = 256;           // ~5 s of slack if Python falls behind
const int MAX_BATCH = 50;           // samples per read_batch() reply

struct Sample {
    uint32_t t_ms;
    int16_t v[6];                   // raw counts: ax, ay, az, gx, gy, gz
};

// Ring buffer: head is the next slot to write, count is how many are unread.
// The oldest unread sample is count slots behind head.
Sample buf[BUF_SIZE];
int head = 0;
int count = 0;
uint32_t dropped = 0;               // samples overwritten before Python read them

bool found = false;
unsigned long next_sample_ms = 0;

void write_reg(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

// Point at reg, then request n bytes
void request(uint8_t reg, uint8_t n) {
    Wire.beginTransmission(ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom(ADDR, n);
}

// Raw counts: ax, ay, az, temp, gx, gy, gz
void read_raw(int16_t v[7]) {
    request(REG_ACCEL_XOUT_H, 14);
    for (int i = 0; i < 7; i++) {
        uint8_t hi = Wire.read();
        uint8_t lo = Wire.read();
        v[i] = (int16_t)((hi << 8) | lo);
    }
}

String imu_check() {
    if (!found) return "not found at 0x68";
    request(REG_WHO_AM_I, 1);
    return "found, whoami=0x" + String(Wire.read(), HEX);
}

String imu_read() {
    if (!found) return "";
    int16_t v[7];
    read_raw(v);
    String s;
    for (int i = 0; i < 3; i++) s += String(v[i] / ACCEL_LSB, 3) + ",";
    for (int i = 4; i < 7; i++) s += String(v[i] / GYRO_LSB, 1) + (i < 6 ? "," : "");
    return s;
}

// Write a sample at head. If the buffer is full, this overwrites the oldest one.
void push(const Sample &s) {
    buf[head] = s;
    head = (head + 1) % BUF_SIZE;
    if (count < BUF_SIZE) count++;
    else dropped++;
}

// Hand Python the oldest unread samples (up to MAX_BATCH) and mark them read.
String read_batch() {
    int n = min(count, MAX_BATCH);
    int tail = (head - count + BUF_SIZE) % BUF_SIZE;
    String s;
    for (int k = 0; k < n; k++) {
        const Sample &p = buf[(tail + k) % BUF_SIZE];
        s += String(p.t_ms);
        for (int i = 0; i < 3; i++) s += "," + String(p.v[i] / ACCEL_LSB, 3);
        for (int i = 3; i < 6; i++) s += "," + String(p.v[i] / GYRO_LSB, 1);
        s += "\n";
    }
    count -= n;
    return s;
}

uint32_t imu_dropped() {
    return dropped;
}

void setup() {
    Wire.begin();
    Wire.beginTransmission(ADDR);
    found = Wire.endTransmission() == 0;
    if (found) {
        write_reg(REG_PWR_MGMT_1, 0);  // wake from sleep
        write_reg(REG_GYRO_CONFIG, RANGE_MAX);
        write_reg(REG_ACCEL_CONFIG, RANGE_MAX);
    }

    Bridge.begin();
    Bridge.provide_safe("imu_check", imu_check);
    Bridge.provide_safe("imu_read", imu_read);
    // provide_safe runs these in the loop thread, so they never race push()
    Bridge.provide_safe("read_batch", read_batch);
    Bridge.provide_safe("imu_dropped", imu_dropped);
    next_sample_ms = millis();
}

void loop() {
    if (!found || millis() < next_sample_ms) return;
    next_sample_ms += SAMPLE_MS;    // fixed schedule, so timing doesn't drift

    int16_t raw[7];
    read_raw(raw);
    Sample s;
    s.t_ms = millis();
    for (int i = 0; i < 3; i++) s.v[i] = raw[i];
    for (int i = 0; i < 3; i++) s.v[3 + i] = raw[4 + i];  // skip temperature
    push(s);
}
