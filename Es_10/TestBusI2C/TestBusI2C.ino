
#include "SdFat.h"
#include <Wire.h>

// --- CONFIGURAZIONE ---
#define MPU_ADDR 0x68
#define SD_CS_PIN 10
#define ACC_PIN 3
#define TS_BUF_SIZE 1024 
#define LOG_SIZE 102400 

struct DataPacket {
  uint8_t data[6];
  uint32_t timeStamp;
}__attribute__((packed));

// --- VARIABILI GLOBALI ---
volatile uint32_t tsCircularBuffer[TS_BUF_SIZE];
volatile uint16_t tsWriteIdx = 0;
volatile int16_t samplesAvailable = 0; // Usiamo int16 per gestire meglio i calcoli
volatile uint32_t totalInterrupts = 0;
volatile uint32_t lostSamplesCounter = 0;

uint16_t tsReadIdx = 0;
uint32_t totalSamplesSaved = 0;
DataPacket sdBuffer[200]; 

SdFat sd;
SdFile file;

// --- ISR ---
void dataReadyISR() {
  uint16_t nextIdx = (tsWriteIdx + 1) % TS_BUF_SIZE;
  
  if (nextIdx == tsReadIdx) {
    lostSamplesCounter++; // Buffer pieno: perdiamo questo campione
  } else {
    tsCircularBuffer[tsWriteIdx] = micros();
    tsWriteIdx = nextIdx;
    samplesAvailable++;
  }
  totalInterrupts++;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  initHardware();

  if (!sd.begin(SD_CS_PIN, SD_SCK_MHZ(24))) {
    Serial.println(F("ERRORE SD")); while(1);
  }
  file.open("DATA_LOG.BIN", O_RDWR | O_CREAT | O_TRUNC);
  file.preAllocate(LOG_SIZE);

  pinMode(ACC_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ACC_PIN), dataReadyISR, RISING);

  Serial.println(F("T_ms\tLetti\tFIFO\tBuffer\tStato"));
}

void loop() {
  static uint32_t lastReadTime = millis();

  // Controllo timeout: se non leggiamo nulla da 200ms, il bus è bloccato
  if (millis() - lastReadTime > 500) {
    handleEmergencyReset();
    lastReadTime = millis();
  }

  if (samplesAvailable >= 64) {
    uint16_t bytesInFifo = getValidFifoCount();
    uint16_t totalToRead = bytesInFifo / 6;

    if (totalToRead > 0) {
      lastReadTime = millis();
      processRead(totalToRead);
      
      // Diagnostica
      float fillPct = (samplesAvailable / (float)TS_BUF_SIZE) * 100.0;
      //Serial.print(millis() % 1000); Serial.print(F("\t"));
      //Serial.print(totalToRead);     Serial.print(F("\t"));
      //Serial.print(bytesInFifo);    Serial.print(F("\t"));
      //Serial.print(fillPct, 1);      Serial.println(F("%\tOK"));
    }

    if (file.curPosition() >= LOG_SIZE) stopLogging();
  }
}

// --- FUNZIONI DI SUPPORTO ---

void initHardware() {
  Wire.begin();
  Wire.setClock(50000); 
  writeReg(0x6B, 0x80); delay(100); // Reset MPU
  writeReg(0x6B, 0x01);             // Wake up
  writeReg(0x19, 0x04);             // 200Hz
  writeReg(0x1A, 0x03);             // DLPF
  writeReg(0x38, 0x01);             // Enable Interrupt
  writeReg(0x23, 0x08);             // FIFO Accel
  writeReg(0x6A, 0x40);             // FIFO Enable
}

void handleEmergencyReset() {
  Serial.println(F("!!! RESET EMERGENZA !!!"));
  initHardware();
  noInterrupts();
  tsReadIdx = tsWriteIdx; 
  samplesAvailable = 0;
  interrupts();
}

uint16_t getValidFifoCount() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x72);
  if (Wire.endTransmission(false) != 0) return 0;
  Wire.requestFrom(MPU_ADDR, 2);
  if (Wire.available() < 2) return 0;
  uint16_t count = (uint16_t)(Wire.read() << 8 | Wire.read());
  return (count > 1024) ? 0 : count;
}

void processRead(uint16_t count) {
  if (count > 200) count = 200;
  
  for (uint16_t i = 0; i < count; i++) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x74);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6);
    for (uint8_t b = 0; b < 6; b++) sdBuffer[i].data[b] = Wire.read();
    
    sdBuffer[i].timeStamp = tsCircularBuffer[tsReadIdx];
    tsReadIdx = (tsReadIdx + 1) % TS_BUF_SIZE;
  }

  file.write((const uint8_t*)sdBuffer, count * sizeof(DataPacket));
  totalSamplesSaved += count;
  
  noInterrupts();
  samplesAvailable -= count;
  interrupts();
}

void stopLogging() {
  detachInterrupt(digitalPinToInterrupt(ACC_PIN));
  file.close();
  Serial.println(F("\n--- TEST FINITO ---"));
  Serial.print(F("Campioni salvati: ")); Serial.println(totalSamplesSaved);
  Serial.print(F("Interrupt totali: ")); Serial.println(totalInterrupts);
  Serial.print(F("Stima campioni persi: ")); Serial.println(totalInterrupts - totalSamplesSaved);
  while(1);
}

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}
