
#include "SdFat.h"

// Configurazione SD e File
#define SD_CS_PIN 10
#define LOG_SIZE 10 * 1024 * 1024 // Pre-allochiamo 10 MB per il test
#define BUFFER_SIZE 1024          // 128 campioni * 8 byte (6 dati + 2 timestamp approx)

SdFat sd;
SdFile file;

uint8_t dummyBuffer[BUFFER_SIZE];
uint32_t writeCount = 0;
const uint32_t TOTAL_BLOCKS = 1000; // Scriveremo circa 1MB di dati

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println(F("--- SD STRESS TEST (SdFat + PreAllocate) ---"));

  // 1. Inizializzazione SD (24MHz è ottimo per R4)
  if (!sd.begin(SD_CS_PIN, SD_SCK_MHZ(24))) {
    Serial.println(F("Errore: SD non inizializzata!"));
    return;
  }

  // 2. Creazione e apertura file
  if (!file.open("stress.bin", O_RDWR | O_CREAT | O_TRUNC)) {
    Serial.println(F("Errore: Apertura file fallita!"));
    return;
  }

  // 3. PRE-ALLOCAZIONE
  Serial.print(F("Allocazione di ")); Serial.print(LOG_SIZE / 1024); Serial.println(F(" KB..."));
  if (!file.preAllocate(LOG_SIZE)) {
    Serial.println(F("Errore: Pre-allocazione fallita! SD frammentata o piena."));
    file.close();
    return;
  }
  Serial.println(F("Spazio riservato correttamente."));

  // Riempio il buffer con dati finti
  for (int i = 0; i < BUFFER_SIZE; i++) dummyBuffer[i] = i % 256;

  // 4. LOOP DI SCRITTURA (Simula il logging)
  uint32_t maxLat = 0;
  uint32_t startTest = millis();

  Serial.println(F("Scrittura in corso..."));
  for (uint32_t b = 0; b < TOTAL_BLOCKS; b++) {
    uint32_t t0 = micros();
    
    size_t written = file.write(dummyBuffer, BUFFER_SIZE);
    
    uint32_t t1 = micros();
    uint32_t lat = t1 - t0;

    if (lat > maxLat) maxLat = lat;
    if (written != BUFFER_SIZE) Serial.println(F("Errore di scrittura!"));
    
    if (b % 100 == 0) Serial.print("."); 
  }

  // 5. TRUNCATE E CHIUSURA
  // Calcoliamo la posizione attuale per tagliare il file alla dimensione reale
  uint32_t finalPos = file.curPosition();
  
  Serial.println(F("\nFase finale: Truncate e Close..."));
  if (!file.truncate(finalPos)) {
    Serial.println(F("Errore: Truncate fallito!"));
  }
  
  file.close();

  // 6. RISULTATI
  uint32_t totalTime = millis() - startTest;
  Serial.println(F("--- RISULTATI FINALI ---"));
  Serial.print(F("Tempo totale: ")); Serial.print(totalTime); Serial.println(F(" ms"));
  Serial.print(F("Latenza MASSIMA: ")); Serial.print(maxLat); Serial.println(F(" us"));
  Serial.print(F("Dimensione finale file: ")); Serial.print(finalPos); Serial.println(F(" byte"));

  if (maxLat < 10000) {
    Serial.println(F("STATO: ECCELLENTE. La pre-allocazione sta funzionando."));
  } else if (maxLat < 100000) {
    Serial.println(F("STATO: BUONO. Compatibile con 200Hz."));
  } else {
    Serial.println(F("STATO: RISCHIOSO. Latenza troppo alta per la FIFO dell'MPU."));
  }
}

void loop() {}

