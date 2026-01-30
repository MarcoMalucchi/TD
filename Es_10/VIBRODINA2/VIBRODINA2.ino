#include "Arduino.h"
#include "SPI.h"
#include "FspTimer.h" //già caricata da pwm.h
#include "pwm.h"
#include "SdFat.h"
#include "Wire.h"

#define STEP_PIN 5   // Step Pin
#define DIR_PIN 6    // Direzione rotazione
#define ACC_PIN 3    // Interrupt accelerometro 
#define SENSOR_PIN 2 // Sensore IR feedback
#define ENABLE_PIN 4 // Attiva motore
#define HOLES 10     // Fori disco controllo 
#define STEPS_4_REV 400.0 // Passi per giro (modificabile con ponticelli on board)
#define DEBUG false
#define F_CLK   48000000UL
#define MAX_TIME 45UL       //mssima durata acquisizione
#define TS_BUF_SIZE 1024

#define MPU_ADDR 0x68   //indirizzo accelerometro
#define BUF_LEN 512    //Dimensione buffer - Multiplo di 512 e 64 dopo aver letto 8 volte FIFO -> buffer RAM pieno
#define BLOCK_LEN 64   //Dati FIFO
#define READFREQ_HZ 10 //Frequenza lettura accelerometro


// --- Timer GPT ---
FspTimer stepTimer;        // genera PWM per STEP quando cambia velocità
FspTimer rampTimer;        // gestisce la rampa di frequenza
PwmOut pwmTimer(STEP_PIN); // genera PWM per STEP a velocità costante

const uint8_t SD_CS_PIN = 10;  //Pin scheda SD

const float MAX_ACC  =  3.00; //rad/sec^2
const float MIN_FREQ =  0.30; //Hz
const float MAX_FREQ = 12.00; //Hz
const int   rampIntervalMs = 50; // ms, intervallo di aggiornamento rampa (dt)
int j = 0, i = 0; //contatori
float targetFreq, currentFreq, freqRev = 0.0;
float countFreq[5] = {0};
int revolution = 0;
float Average = 0, Sum = 0;
volatile bool pinLevel = LOW;

volatile float targetStepFreq, currentStepFreq, rampStepAcc;
volatile int hole_counter = 0; // --- PID (su RPM) ---
volatile unsigned long t_start = 0;
volatile unsigned long t_hole = 0;
unsigned long revPeriodMicros = 0;
unsigned int freq_prescaler = 0;
uint32_t sincroTime = 0;

uint16_t bufIndex = 0;
volatile bool fifoFlag = false;  //diventa true quando IL FIFO è pieno durante logging
volatile bool loggerStop = false; //diventa true quando si da il comando LOGGER_OFF
bool firstPass = false;
volatile uint16_t count = 0;
volatile uint16_t TimeStampOffset = 0;
volatile uint32_t lastTimestamp = 0;
volatile uint16_t block = 0;
uint32_t startLoggedTime;
uint32_t timeBaseMillis;

struct system {
  unsigned int motorOn: 1;
  unsigned int motorDir: 1;
  unsigned int dataLogger: 1;
  unsigned int freqChanged: 1;
  unsigned int motorChanged: 1;
  unsigned int Acc: 1;
  unsigned int feedback: 1;
  unsigned int gateway: 1;
} process;

const size_t CMD_SIZE = 16;
char cmd[CMD_SIZE];
byte idx = 0;

struct __attribute__((packed)) packet{
  uint16_t syncWord = 0xAAAA; // The "Sync Word" (2 bytes)  
  uint8_t data[6];
  uint32_t timeStamp;
};

packet dataBuf[BUF_LEN];
volatile uint32_t timeStamp[2][BLOCK_LEN];
volatile uint8_t activeBuff = 0;

// Per SD
SdFat sd;
SdFile dataFile;

volatile uint32_t tsCircularBuffer[1024]; // Buffer circolare timestamp
volatile uint16_t tsWriteIdx = 0;
volatile int16_t samplesAvailable = 0;
volatile uint16_t tsReadIdx = 0;


// Per date time SD
uint16_t fatYear  = 2026;
uint8_t  fatMonth = 1;
uint8_t  fatDay   = 21;

uint8_t  fatHour  = 14;
uint8_t  fatMin   = 38;
uint8_t  fatSec   = 0;   // FAT risoluzione 2 secondi

const uint8_t daysInMonth[] = {
  31, 28, 31, 30, 31, 30,
  31, 31, 30, 31, 30, 31
};

// --- prototipi ---
bool initSD(uint8_t csPin, uint32_t clockMHz = 25); //clockMHz predefinito 25
void setupMPU(void);
void writeMPU(uint8_t reg, uint8_t val);
uint16_t readFIFOCount(void);
int smartBegin(float freq);
void onIrPulse(void);
void onStepTimer(timer_callback_args_t *args);
void dataReadyAcc(void);
void onRampTimer(timer_callback_args_t *args);
void resetValue(void);
void handleCommand(char *cmd);
bool isFloat(const char *s);
int findNextDataFile(const char *folder, char *outName, size_t outSize, float freq);
void floatToString(float value, char *buf, size_t size, char sep);
int smartPrescaler(float freq);
void setupMPU(void);
uint16_t getValidFifoCount(void);
void handleEmergencyReset(void);
bool initSD(uint8_t csPin, uint32_t clockMHz);
inline void setBit(char *arr, int index);
inline void clearBit(char *arr, int index);
inline bool readBit(const char *arr, int index);
void printLastDetection(void);
void dateTime(uint16_t* date, uint16_t* time);

// reset variabili
void resetValue(void) {
  digitalWrite(ENABLE_PIN, HIGH);
  targetFreq = MIN_FREQ;
  currentFreq = 0.0;
  currentStepFreq = 0.0;
  process.motorOn = false;
  process.dataLogger = false;
  process.freqChanged = true;
  process.motorChanged = false;
  process.Acc = false;
  process.feedback = false;
  process.gateway = false;
  stepTimer.stop();
  rampTimer.stop();
  pwmTimer.end();
  block = 0;
}

// --- ISR sensore IR ---
void onIrPulse(void) {
  t_hole = micros();
  process.gateway = true;
}

// --- ISR accelerometro ---
void dataReadyAcc() {
  uint16_t next = (tsWriteIdx + 1) % 1024;
  if (next != tsReadIdx) {
    tsCircularBuffer[tsWriteIdx] = micros();
    tsWriteIdx = next;
    samplesAvailable++;
  }
}
// --- Rampa: applica incremento o decremento rispettando alpha_acc/alpha_dec ---
void onRampTimer(timer_callback_args_t *args) {
  if (!process.motorOn) return; //Se stepTimer non è attivo
  currentStepFreq += rampStepAcc;
  if ((currentStepFreq - targetStepFreq) * rampStepAcc >= 0.0) currentStepFreq = targetStepFreq;
  stepTimer.set_frequency(currentStepFreq * 2.0);
}

// --- Rampa: applica incremento o decremento rispettando alpha_acc/alpha_dec ---
void onStepTimer(timer_callback_args_t *args) {
  pinLevel = !pinLevel;
  digitalWrite(STEP_PIN, pinLevel);
}

uint16_t getValidFifoCount(void) {
  Wire.beginTransmission(0x68);
  Wire.write(0x72); // Registro FIFO_COUNT
  if (Wire.endTransmission(false) != 0) return 0;
  
  Wire.requestFrom(0x68, 2);
  if (Wire.available() < 2) return 0;
  
  uint16_t count = (uint16_t)(Wire.read() << 8 | Wire.read());
  
  // Protezione anti-crash: la FIFO dell'MPU6050 è di 1024 byte.
  // Se leggiamo di più, il bus è disturbato.
  if (count > 1024) {
    return 0; 
  }
  return count;
}

void handleEmergencyReset() {
  if (DEBUG) Serial.println(F("!!! RESET HW SENSITORE !!!"));
  
  // Re-inizializzazione bus e sensore
  Wire.begin();
  Wire.setClock(50000);
  
  writeMPU(0x6B, 0x80); // Reset totale MPU
  delay(50);
  writeMPU(0x6B, 0x01); // Sveglia
  writeMPU(0x19, 0x04); // Torna a 200Hz
  writeMPU(0x1A, 0x03); // Filtro DLPF
  writeMPU(0x38, 0x01); // Riattiva Interrupt
  writeMPU(0x23, 0x08); // FIFO solo accelerometro
  writeMPU(0x6A, 0x44); // Reset e Enable FIFO
  
  // Sincronizzazione indici software
  noInterrupts();
  tsReadIdx = tsWriteIdx; // Svuota virtualmente il buffer timestamp
  samplesAvailable = 0;
  interrupts();
}


//******************************************************
//SETUP
//******************************************************

void setup() {
  Serial.begin(115200); while (!Serial);
  pinMode(STEP_PIN,   OUTPUT);
  pinMode(DIR_PIN,    OUTPUT);
  pinMode(SENSOR_PIN, INPUT_PULLUP); // sensore IR
  pinMode(ACC_PIN,    INPUT); // interrupt accelerometro
  pinMode(ENABLE_PIN, OUTPUT);

  process.motorOn = false; //Motore spento
  process.motorDir = HIGH; //Rotazione fissa antioraria (attenzione ai collegamenti del motore)
  process.dataLogger = false; //Data Logger OFF
  targetFreq = MIN_FREQ;
  targetStepFreq = MIN_FREQ * STEPS_4_REV;
  currentStepFreq = 0.0;
  currentFreq = 0.0;
  digitalWrite(DIR_PIN, process.motorDir); // direzione fissa
  digitalWrite(ENABLE_PIN, HIGH); // Motore spento

  //pwmTimer.begin(0.0f, 0.0f);
  //pwmTimer.suspend();

  // Timer per rampa
  bool ok = rampTimer.begin(TIMER_MODE_PERIODIC, GPT_TIMER, 4, 20.0, 0, onRampTimer);
  if (!ok) {
    Serial.println("Errore inizializzazione ramp timer!");
    while (1);
  }
  ok = stepTimer.begin(TIMER_MODE_PERIODIC, GPT_TIMER, 5, 1.0, 0, onStepTimer);
  if (!ok) {
    Serial.println("Errore inizializzazione step timer!");
    while (1);
  }

  // abilita l'interrupt overflow con priorità 13 e 12
  stepTimer.setup_overflow_irq(12); //prima step
  rampTimer.setup_overflow_irq(13);
  rampTimer.open();
  stepTimer.open();

  // Interrupt sensore IR
  attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), onIrPulse, RISING);

  if (!initSD(SD_CS_PIN)) {
    Serial.println("Errore inizializzazione SD!");
    while (1);
  }

  SdFile::dateTimeCallback(dateTime); //Abilita data e ora per i file

  Wire.begin();
  Wire.setClock(100000);
  setupMPU();


  Serial.println(F("Parser pronto (max 3 decimali)."));
  Serial.println(F("Comandi: +, ++, +++, -, --, ---, +n, -n, n"));
  Serial.println(F("GO, STOP, LOGGER_ON, LOGGER_OFF, LIST, RESET, FREQ, P, HELP, ?"));
  Serial.println(F("================================================================================"));
}

void loop() {
  static float inFreq;
  static unsigned long startAcc, endAcc;
  static int targetTick, currentTick;

  if (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { // fine riga
      if (idx > 0) {
        cmd[idx] = '\0';
        handleCommand(cmd); //Elabora il comando
        idx = 0;
      }
    }
    else if (idx < CMD_SIZE - 1) {
      cmd[idx++] = toupper(c); //Tutto maiuscolo
    }
  } //legge un carattere alla volta ma fino alla fine non fa niente

  if (process.motorChanged) {
    process.motorChanged = false;
    if (!process.motorOn) {
      resetValue();
      return;
    } else {
      Serial.println("Motore START");
      digitalWrite(ENABLE_PIN, LOW); //Abilita il motore
      stepTimer.start();
      process.freqChanged = true;
    }
  }

  if (process.freqChanged) {
    Serial.println(F("busy"));
    detachInterrupt(digitalPinToInterrupt(SENSOR_PIN)); //disabilita il controllo feedback
    hole_counter = 0;
    Sum = 0; Average = 0, revolution = 0; t_start = 0;
    memset(countFreq, 0, sizeof(countFreq));
    float acc = (targetFreq > currentFreq ? MAX_ACC : -MAX_ACC); //Calcolo accelerazione
    targetStepFreq = targetFreq * STEPS_4_REV;
    rampStepAcc = acc * STEPS_4_REV * rampIntervalMs / 2000 / PI;
    process.freqChanged = false;
    process.Acc = true;
    process.feedback = false;

    // Prepara il timer PWM
    pwmTimer.end(); //Attenzione, forse non serve

    R_PFS->PORT[4].PIN[5].PmnPFS_b.PMR = 0;  // Disattiva funzione periferica (PWM, UART, ecc.)
    R_PFS->PORT[4].PIN[5].PmnPFS_b.PSEL = 0; // Seleziona funzione GPIO (non periferica)
    // R_PORT4->PDR_b.PDR5 = 1; // Imposta direzione: 1 = output
    pinMode(STEP_PIN, OUTPUT);

    stepTimer.start();
    rampTimer.start();

    startAcc = millis();
    inFreq = currentFreq;
    Serial.print(F("Target frequency:  ")); Serial.print(targetFreq,  3); Serial.println(F(" Hz"));
    Serial.print(F("Current frequency: ")); Serial.print(currentFreq, 3); Serial.println(F(" Hz"));
    Serial.print(F("Step acceleration: ")); Serial.print(rampStepAcc, 3); Serial.println(F(" rad/sec^2"));
  }

  //Accelerazione finita
  if (((currentStepFreq - targetStepFreq) * rampStepAcc >= 0.0) && process.Acc) {
    stepTimer.stop();

    freq_prescaler = smartBegin(targetStepFreq);

    pwmTimer.resume();
    rampTimer.stop();
    endAcc = millis();
    targetTick = F_CLK / freq_prescaler / targetStepFreq;
    process.Acc = false;
    float realAcc = 0.0;
    currentFreq = targetFreq;
    float Dt = (endAcc - startAcc) / 1000.0;
    realAcc = 2 * PI * (currentFreq - inFreq) / Dt;

    Serial.print(F("Step frequency:    ")); Serial.print(targetStepFreq, 3); Serial.println(F(" Hz"));
    Serial.print(F("Real acceleration: ")); Serial.print(realAcc,        3); Serial.println(F(" rad/sec^2"));
    Serial.print(F("Time acceleration: ")); Serial.print(Dt); Serial.println(F(" sec"));
    if (DEBUG) {
      Serial.print(F("freq_prescaler   ")); Serial.println(freq_prescaler);
    }

    attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), onIrPulse, RISING); //abilita il controllo feedback
  }

  if (process.gateway && !process.dataLogger) {
    process.gateway = false;
    if (t_start == 0) {
      t_start = t_hole;
    } else {
      freqRev = round(1.0e9 / (float)(t_hole - t_start)) / 1000.0;
      t_start = t_hole;
      revolution++;

      if (revolution > 1) {
        if (abs((freqRev - Average) * 100 / Average) > 5.0) freqRev = Average;
      } else Average = freqRev;
      Sum -= countFreq[revolution % 5];
      Sum += freqRev;
      Average = round(1000 * Sum / (revolution < 5 ? revolution : 5)) / 1000;
      countFreq[revolution % 5] = freqRev;

      //currentTick = F_CLK/freq_prescaler/(Average*STEPS_4_REV);
      //int dTick = (currentTick-targetTick)*Kp; //=0 ok; <0 veloce >0 lento
      if (targetFreq - Average == 0) {
        if (process.feedback || revolution == 1) {
          Serial.print(F("Frequenza target: ")); Serial.print(targetFreq, 3); Serial.print(F(" Hz; "));
          Serial.print(F("Frequenza media su ")); Serial.print((revolution < 5 ? revolution : 5)); Serial.print(F(" giri: "));
          Serial.print(Average, 3); Serial.print(F(" Hz; "));
          Serial.print(F("raggiunta dopo: ")); Serial.print(revolution); Serial.println(F(" giri"));
          process.feedback = false;
          Serial.println(F("available"));
        }
      } else {
        if (!process.feedback) {
          Serial.print(F("Perso sincronismo al giro: ")); Serial.print(revolution);
          Serial.print(F("; Frequenza misurata: ")); Serial.print(freqRev, 3); Serial.println(F(" Hz; "));
        }
        process.feedback = true;
      }
    }
  }
  if (process.dataLogger) {
    uint32_t tNow = micros();
    uint32_t tLogger = tNow - startLoggedTime;
  
    // 1. Limite temporale di sicurezza
    if (tLogger > 45000000UL) loggerStop = true; 

    // 2. Controllo emergenza: Reset se il sensore non risponde per 250ms
    static uint32_t lastSuccessfulRead = 0;
    if (millis() - lastSuccessfulRead > 500 && !loggerStop) {
      handleEmergencyReset(); 
      lastSuccessfulRead = millis();
    }

    // 3. Logica di lettura FIFO (si attiva quando ci sono abbastanza campioni nel buffer RAM)
    // Nota: 'samplesAvailable' è incrementato dalla tua ISR 'dataReadyAcc'
    if (samplesAvailable >= 64 && !loggerStop) {
    
      uint16_t bytesInFifo = getValidFifoCount(); // Verifica quanti dati ha l'MPU

      uint8_t remainder = bytesInFifo % 6;
      if (remainder > 0) {
        // Throw away the misaligned bytes
        Wire.beginTransmission(0x68);
        Wire.write(0x74);
        Wire.endTransmission(false);
        Wire.requestFrom(0x68, remainder);
        for (uint8_t r = 0; r < remainder; r++) Wire.read();

        // 2. IMPORTANT: Synchronize the Timestamp Buffer
        // A remainder implies the last sample was incomplete/corrupted.
        // We advance the read pointer to drop the corresponding timestamp.
        if (samplesAvailable > 0) {
          tsReadIdx = (tsReadIdx + 1) % TS_BUF_SIZE;
          samplesAvailable--;
        }

        // Update our count after cleaning
        bytesInFifo -= remainder; 
      }      

      uint16_t totalToRead = bytesInFifo / 6;

      // Limitiamo la lettura per non saturare il tuo dataBuf
      if (totalToRead > (BUF_LEN - bufIndex)) totalToRead = BUF_LEN - bufIndex;

      if (totalToRead > 0) {
        lastSuccessfulRead = millis();
      
        // Lettura dei dati dall'MPU e inserimento nel dataBuf
        for (uint16_t i = 0; i < totalToRead; i++) {
          // Lettura burst 6 byte accelerometro
          Wire.beginTransmission(0x68);
          Wire.write(0x74);
          Wire.endTransmission(false);
          Wire.requestFrom(0x68, 6);
          for (uint8_t b = 0; b < 6; b++) {
            dataBuf[bufIndex].data[b] = Wire.read();
          }

          // Recupero timestamp dal buffer circolare della ISR
          dataBuf[bufIndex].timeStamp = tsCircularBuffer[tsReadIdx];
          dataBuf[bufIndex].syncWord = 0xAAAA;
        
          // Avanzamento indici
          tsReadIdx = (tsReadIdx + 1) % TS_BUF_SIZE;
          bufIndex++;
        }

        // Sottrazione campioni letti (Atomica)
        noInterrupts();
        samplesAvailable -= totalToRead;
        interrupts();

        // Se il dataBuf è pieno, scriviamo su SD
        if (bufIndex >= BUF_LEN-1) {
          dataFile.write((const uint8_t*)dataBuf, BUF_LEN * sizeof(packet));
          bufIndex = 0;
          block++; // Incrementa il contatore blocchi per il tuo report finale
        }
      }
    }
  }
  if (loggerStop) {
    dataBuf[bufIndex].timeStamp = sincroTime;
    for (int i = 0; i < 4; i++) dataBuf[bufIndex].data[i] = (startLoggedTime >> (i * 8)) & 0xFF;
    uint16_t deltaT = (micros() - startLoggedTime) / 1000;
    dataBuf[bufIndex].data[4] = deltaT & 0xFF; dataBuf[bufIndex].data[5] = (deltaT >> 8) & 0xFF;
    size_t written = dataFile.write((uint8_t*)dataBuf, (bufIndex + 1) * sizeof(packet));

    detachInterrupt(digitalPinToInterrupt(ACC_PIN)); //Disabilita interrupt

    char availableFile[65]; // Assicurati che sia abbastanza grande per il percorso
    dataFile.getName(availableFile, sizeof(availableFile));

    // TRUNCATE AND CLOSE
    uint32_t finalPos = dataFile.curPosition();

    Serial.println(F("\nFase finale: Truncate e Close..."));
    if (!dataFile.truncate(finalPos)) {
      Serial.println(F("Errore: Truncate fallito!"));
    }

    dataFile.close();

    fifoFlag = false;
    process.dataLogger = false;
    loggerStop = false;

    setupMPU();  // Reset MPU
    tsWriteIdx = tsReadIdx = 0;

    Serial.println(F("Data Logger disattivato."));
    Serial.print(F("Campionati n. ")); Serial.print(block * BUF_LEN + bufIndex); Serial.print(F(" punti.")); //n. dati salvati
    Serial.print(F("nel file: ")); Serial.println(F(availableFile));
    Serial.print(F("Dimensione del file: ")); Serial.print(finalPos); Serial.println(F(" byte"));
    Serial.print(F("Tempo di campionamento ")); Serial.print((micros() - startLoggedTime) / 1000000.0, 3); Serial.println(F(" sec."));
    block = 0;
    Serial.println(F("available"));
  }

}

//==============================================================================
//Parser comandi
//==============================================================================
void handleCommand(char *cmd) {
  if (process.dataLogger) {
    if (strcmp(cmd, "LOGGER_OFF") == 0) {
      loggerStop = true;
    } else {
      Serial.print(F("Data Logger attivo. Comando ")); Serial.print(cmd); Serial.println (F(" non valido o non attivo."));
    }
    return; // esce subito, ignorando il resto
  }
  process.freqChanged = false;
  if      (strcmp(cmd, "+++") == 0) {
    targetFreq += 0.100;
    process.freqChanged = true;
  }
  else if (strcmp(cmd, "++")  == 0) {
    targetFreq += 0.010;
    process.freqChanged = true;
  }
  else if (strcmp(cmd, "+")   == 0) {
    targetFreq += 0.001;
    process.freqChanged = true;
  }
  else if (strcmp(cmd, "---") == 0) {
    targetFreq -= 0.100;
    process.freqChanged = true;
  }
  else if (strcmp(cmd, "--")  == 0) {
    targetFreq -= 0.010;
    process.freqChanged = true;
  }
  else if (strcmp(cmd, "-")   == 0) {
    targetFreq -= 0.001;
    process.freqChanged = true;
  }
  else if ((cmd[0] == '+' || cmd[0] == '-') && isdigit(cmd[1])) { //evita il doppio segno es: ++5
    float delta = atof(cmd);
    if (delta != 0.0f) {
      targetFreq += delta;
      process.freqChanged = true;
    }
  }
  else if (isFloat(cmd)) {
    targetFreq = atof(cmd);
    process.freqChanged = true;
  }
  else if (strcmp(cmd, "GO")   == 0) {
    process.motorOn = true;
    process.motorChanged = true;
  }
  else if (strcmp(cmd, "STOP") == 0) {
    process.motorOn = false;
    process.motorChanged = true;
  }
  else if (strcmp(cmd, "LOGGER_ON")   == 0) {
    char availableFile[64];
    Serial.print("busy");
    if (DEBUG) Serial.println(availableFile);
    if (findNextDataFile("DATA", availableFile, sizeof(availableFile), targetFreq) == 0) {
      if (DEBUG) Serial.println(availableFile);
      //Attivare il data logger
      process.dataLogger = true;
      count = 0;
      memset(dataBuf, 0, sizeof(dataBuf));
      memset((void*)timeStamp, 0, sizeof(timeStamp));
      firstPass = true;

      // Creazione e apertura file
      if (!dataFile.open(availableFile, O_RDWR | O_CREAT | O_TRUNC)) {
        Serial.println(F("Errore: Apertura file fallita!"));
        return;
      }
      // PRE-ALLOCAZIONE

      if (!dataFile.preAllocate(2*READFREQ_HZ * (MAX_TIME + 15)*sizeof(packet))) {
        Serial.println(F("Errore: Pre-allocazione fallita! SD frammentata o piena."));
        dataFile.close();
        return;
      }
      
      writeMPU(0x6A, 0x04); // FIFO reset
      writeMPU(0x23, 0x08); // FIFO_EN: accelerometer only
      writeMPU(0x6A, 0x40); // FIFO enable
      // Interrupt accelerometro
      attachInterrupt(digitalPinToInterrupt(ACC_PIN), dataReadyAcc, RISING);
      startLoggedTime = micros();
      sincroTime = t_start; //tempo dell'ultimo passaggio buco (opposto alla massa rotante)

    } else Serial.println("Impossibile attivare data logger. Controlla SD"); //volendo si può valutare errore
  }
  else if (strcmp(cmd, "OFF")  == 0) {
    process.dataLogger = false;
  }
  else if (strcmp(cmd, "RESET") == 0) {
    resetValue();
  }
  else if (strcmp(cmd, "LIST") == 0) {
    Serial.print(F("Fequenza impostata: ")); Serial.println(currentFreq, 3);
    Serial.print(F("Fequenza media su 5 giri: ")); Serial.println(Average, 3);
    Serial.print(F("Motore: ")); Serial.println(process.motorOn ? F("ON") : F("OFF"));
    Serial.print(F("Data Logger: ")); Serial.println(process.dataLogger ? F("ATTIVO") : F("FERMO"));
  }
  else if (strcmp(cmd, "FREQ") == 0) {
    Serial.print(F("Fequenza impostata: ")); Serial.println(currentFreq, 3);
    Serial.print(F("Fequenza misurata media su 5 giri: ")); Serial.println(Average, 3);
  }
  else if (strncmp(cmd, "TIME", 4) == 0) {
    setTimeFromCommand(cmd);
    Serial.print(F("Data impostata: ")); Serial.println(cmd + 4);
  }

  //else if (strcmp(cmd, "HELP")  == 0 || (strcmp(cmd, "?")  == 0)) { HelpCommand(); }
  else if (strcmp(cmd, "PRINT")  == 0) {
    printLastDetection();
  }
  else {
    Serial.print(F("Comando sconosciuto: ")); Serial.println(cmd);
  }

  if (process.freqChanged) {
    targetFreq = constrain(roundf(targetFreq * 1000.0f) / 1000.0, 0.3, 12.0); // arrotondamento a 3 decimali
    Serial.print(F("Target frequency: ")); Serial.print(targetFreq, 3); Serial.println(F(" Hz"));
  }
}

//**********************************************
// verifica se una stringa è un float valido
//**********************************************

bool isFloat(const char *s) {
  bool pointSeen = false;
  if (*s == '\0') return false;
  for (; *s; s++) {
    if (isdigit(*s)) continue;
    if (*s == '.' && !pointSeen) {
      pointSeen = true; continue;
    } return false;
  } return true;
}

int smartBegin(float freq) {
  int prescaler;

  // Spegne e resetta il timer per permettere una nuova selezione del prescaler
  pwmTimer.end();

  if (freq < 280) {
    pwmTimer.begin(44.0f, 50.0f);
    //pwmTimer.suspend();
    prescaler = 64;
  }
  else if (freq < 1200) {
    pwmTimer.begin(185.0f, 50.0f);
    //pwmTimer.suspend();
    prescaler = 4;
  }
  else {
    pwmTimer.begin(4000.0f, 50.0f);
    //pwmTimer.suspend();
    prescaler = 1;
  }
  pwmTimer.suspend();
  unsigned int tick = 48000000UL / prescaler / freq;
  pwmTimer.period_raw(tick);
  pwmTimer.pulseWidth_raw(tick / 2);
  return prescaler;
}

int smartPrescaler(float freq) {
  if (freq < 280.0) return 64;
  else if (freq < 1200.0) return 4;
  else return 1;

}

//==============================================================================
//Gestione GY-521
//==============================================================================

// Scrive un valore nel registro
void writeMPU(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

// Setup GY-521
void setupMPU() {
  writeMPU(0x6B, 0x80);  delay(50);// reset
  writeMPU(0x6B, 0x01);  delay(10);// wake up

  writeMPU(0x1A, 0x03); // DLPF 44Hz
  writeMPU(0x19, (uint8_t)(1000 / READFREQ_HZ - 1)); // sample rate
  writeMPU(0x1C, 0x00); // ±2g
  writeMPU(0x1B, 0x00); // gyro ±250°/s, se serve

  // reset FIFO then enable FIFO accel only
  writeMPU(0x37, 0x00); // INT_PIN_CFG: active high, push-pull
  writeMPU(0x38, 0x01); // INT_ENABLE: data ready
  writeMPU(0x6A, 0x00); // FIFO disabilitato
  writeMPU(0x6A, 0x04); // FIFO reset
  delay(1);
  writeMPU(0x23, 0x08); // FIFO_EN: accelerometer only
}

// FIFO count
uint16_t readFIFOCount() {
  uint16_t c = 0;
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x72); // FIFO_COUNT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)2);
  if (Wire.available() >= 2) {
    c = (Wire.read() << 8) | Wire.read();
  }
  return c;
}

//==============================================================================
//Fine gestione GY-521
//==============================================================================


//==============================================================================
//Gestione microSD
//==============================================================================

// Inizializza la SD
bool initSD(uint8_t csPin, uint32_t clockMHz) {
  if (!sd.begin(csPin, SD_SCK_MHZ(clockMHz))) {
    return false;
  }
  return true;
}

// Trova o crea la cartella e restituisce il primo file libero
int findNextDataFile(const char *folder, char *outName, size_t outSize, float freq) {
  // Se la cartella non esiste, crea
  if (!sd.exists(folder)) {
    if (!sd.mkdir(folder)) {
      return 1;  // errore creazione cartella
    }
  }

  // Apri la cartella
  File dir = sd.open(folder);
  if (!dir || !dir.isDirectory()) {
    return 2;  // errore apertura cartella
  }

  char used[125] = {0};
  char name[64];

  // Scansione file nella cartella
  while (true) {
    File entry = dir.openNextFile();
    if (!entry) break;

    if (!entry.isDirectory()) {
      entry.getName(name, sizeof(name));

      if (strncmp(name, "DATA", 4) == 0 && strlen(name) >= 7) {
        char numBuf[4] = {0};
        strncpy(numBuf, name + 4, 3);
        char *endptr;
        long val = strtol(numBuf, &endptr, 10);

        // Se sono proprio 3 cifre valide used è un vettore di 1000 bit (125*8) 0->numero libero, 1->occupato
        if (endptr == numBuf + 3) {
          setBit(used, val);
        }
      }
    }
    entry.close();
  }
  dir.close();

  // Trova il primo numero libero e scrivilo nel buffer
  for (int i = 0; i < 1000; i++) {
    if (!readBit(used, i)) {
      char buff[8];
      floatToString(freq, buff, sizeof(buff), '_');
      snprintf(outName, outSize, "%s/DATA%03d_f%s.bin", folder, i, buff);
      return 0;  // OK
    }
  }

  return 3;  // Nessun nome libero
}

// Converte float in string sostituendo il separatore
void floatToString(float value, char *buf, size_t size, char sep) {
  dtostrf(value, 1, 3, buf);
  for (size_t i = 0; i < size && buf[i]; i++) {
    if (buf[i] == '.') {
      buf[i] = sep;
      break;
    }
  }
}

// --- Gestione array di bit (8 bit per char) ---
inline void setBit(char *arr, int index) {
  arr[index / 8] |=  (1 << (index % 8));
}

inline void clearBit(char *arr, int index) {
  arr[index / 8] &= ~(1 << (index % 8));
}

inline bool readBit(const char *arr, int index) {
  return arr[index / 8] &  (1 << (index % 8));
}

//==============================================================================
//Fine gestione microSD
//==============================================================================
/*
  //==============================================================================
  //Help command
  //==============================================================================
  void HelpCommand (void) {
  Serial.println(F("+/-     -> Aggiunge/toglie 0.001 Hz dalla frequenza corrente"));
  Serial.println(F("++/--   -> Aggiunge/toglie 0.010 Hz dalla frequenza corrente"));
  Serial.println(F("+++/--- -> Aggiunge/toglie 0.100 Hz dalla frequenza corrente"));
  Serial.println(F("+n/-n   -> Aggiunge/toglie n Hz dalla frequenza corrente"));
  Serial.println(F("n       -> Imposta ad n la frequenza corrente"));
  Serial.println(F("GO/STOP -> Avvia/ferma il motore"));
	Serial.println(F("LIST    -> Stampa sul monitor seriale report di funzionamento"));
	Serial.println(F("RESET   -> Reset"));
	Serial.println(F("FREQ    -> Stampa sul monitor seriale la frequenza impostata e l'ultima rilevata"));
	Serial.println(F("PRINT   -> Stampa sul monitor seriale l'ultima acquisizione"));
	Serial.println(F("?/HELP  -> Stampa sul monitor seriale l'elenco dei comandi"));
  }
*/
//==============================================================================
//Print last detection
//==============================================================================
void printLastDetection(void) {
  SdFile dir;
  SdFile file;

  if (!dir.open("DATA")) {
    Serial.println(F("Cartella /DATA non trovata!"));
    return;
  }

  char filename[32];
  char latestName[32] = "";
  int maxIndex = -1;

  // Scansiona i file nella cartella /DATA
  while (file.openNext(&dir, O_READ)) {
    file.getName(filename, sizeof(filename));

    if (!file.isDir() && strncmp(filename, "DATA", 4) == 0 && strstr(filename, ".bin")) {
      // Estrai le tre cifre nnn da "DATAnnnxxxxx.bin"
      char nnnStr[4];
      strncpy(nnnStr, filename + 4, 3);
      nnnStr[3] = '\0';
      int nnn = atoi(nnnStr);

      if (nnn > maxIndex) {
        maxIndex = nnn;
        strcpy(latestName, filename);
      }
    }
    file.close();
  }
  dir.close();

  if (maxIndex < 0) {
    Serial.println(F("Nessun file trovato in /DATA!"));
    return;
  }

  // Costruisci percorso completo
  char fullPath[40];
  snprintf(fullPath, sizeof(fullPath), "/DATA/%s", latestName);

  Serial.print("Ultimo file: ");
  Serial.println(fullPath);

  if (!file.open(fullPath, O_READ)) {
    Serial.println(F("Errore apertura file!"));
    return;
  }

  packet s;
  uint32_t countRecord = 0;

  // Leggi e stampa tutti i record
  while (file.read(&s, sizeof(packet)) == sizeof(packet)) {
    int16_t x = ((int16_t)s.data[0] << 8) | s.data[1];
    int16_t y = ((int16_t)s.data[2] << 8) | s.data[3];
    int16_t z = ((int16_t)s.data[4] << 8) | s.data[5];

    Serial.print("Record ");
    Serial.print(countRecord++);
    Serial.print(": t=");
    Serial.print(s.timeStamp);
    Serial.print("  X=");
    Serial.print(x);
    Serial.print("  Y=");
    Serial.print(y);
    Serial.print("  Z=");
    Serial.println(z);
  }

  Serial.print("Totale record letti: ");
  Serial.println(countRecord);

  file.close();
}

//********************************************************
// Regola Data e Ora per SdFat
//********************************************************
void dateTime(uint16_t* date, uint16_t* time) {
  uint32_t elapsed = (millis() - timeBaseMillis) / 1000;

  uint32_t sec  = fatSec  + elapsed;
  uint32_t min  = fatMin  + sec / 60;
  uint32_t hour = fatHour + min / 60;
  uint32_t day  = fatDay  + hour / 24;

  sec  %= 60;
  min  %= 60;
  hour %= 24;

  fatDay = day;
  normalizeDate();

  *date = FAT_DATE(fatYear, fatMonth, fatDay);
  *time = FAT_TIME(hour, min, sec);
}

//******************************************************
//Regola la data per cambio giorno
//******************************************************

void normalizeDate(void) {
  uint8_t dim = daysInMonth[fatMonth - 1];

  // febbraio bisestile
  if (fatMonth == 2 && (fatYear % 4 == 0))
    dim = 29;

  if (fatDay > dim) {
    fatDay = 1;
    fatMonth++;
  }

  if (fatMonth > 12) {
    fatMonth = 1;
    fatYear++;
  }
}

//********************************************************
// estrae data da comando TIME nel formato TIME AAAA/MM/GG HH:MM:SS
//********************************************************

bool setTimeFromCommand(const char* cmd) {
  uint8_t i = 0;

  // cerca primo numero (anno)
  while (cmd[i] && !isdigit(cmd[i])) i++;
  if (!cmd[i]) return false;

  fatYear  = atoi(cmd + i);       // AAAA
  fatMonth = atoi(cmd + i + 5);   // MM
  fatDay   = atoi(cmd + i + 8);   // GG

  // salta alla parte ora
  i += 10;  // AAAA/MM/GG
  while (cmd[i] && !isdigit(cmd[i])) i++;
  if (!cmd[i]) return false;

  fatHour = atoi(cmd + i);        // HH
  fatMin  = atoi(cmd + i + 3);    // MM
  fatSec  = atoi(cmd + i + 6);    // SS

  if (fatMonth < 1 || fatMonth > 12) return false;
  if (fatDay   < 1 || fatDay   > 31) return false;
  if (fatHour  > 23) return false;
  if (fatMin   > 59) return false;
  if (fatSec   > 59) return false;

  normalizeDate(); //mette a posto data e ora
  timeBaseMillis = millis();
  Serial.print(F("Data AAAA/MM/GG: "));
  Serial.print(fatYear); Serial.print(F("/"));
  Serial.print(fatMonth); Serial.print(F("/"));
  Serial.print(fatDay); Serial.print(F(" Ora HH:MM:SS "));
  Serial.print(fatHour); Serial.print(F(":"));
  Serial.print(fatMin); Serial.print(F(":"));
  Serial.println(fatSec);

  return true;
}


void readFullFifo(void) {
  const uint8_t samplesPerBlock = 4;
  const uint8_t totalSamples = 128;
  const uint8_t iterations = totalSamples / samplesPerBlock; // 32

  // Puntiamo al registro FIFO una sola volta prima del ciclo
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x74);
  Wire.endTransmission(false);

  for (uint8_t i = 0; i < iterations; i++) {
    // Richiediamo 24 byte (4 campioni)
    if (Wire.requestFrom(MPU_ADDR, (uint8_t)24) != 24) continue;

    for (uint8_t j = 0; j < samplesPerBlock; j++) {
      // k serve solo per pescare il timestamp corretto (0-127)
      uint8_t k = (i * 4) + j;

      // Verifichiamo di non eccedere la dimensione di 1024
      if (bufIndex < 1024) {
        // Assegniamo il timestamp dal buffer di interrupt
        dataBuf[bufIndex].timeStamp = timeStamp[activeBuff ^ 1][k];

        // Leggiamo i 6 byte dell'accelerazione
        Wire.readBytes(dataBuf[bufIndex].data, 6);

        bufIndex++; // Incrementiamo l'indice globale
      }
    }
  }
}
