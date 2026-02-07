#include "Arduino.h"
#include "SPI.h"
#include "FspTimer.h" //già caricata da pwm.h
#include "pwm.h"
#include "SdFat.h"
#include "Wire.h"

#define STEP_PIN 5   // Step Pin
#define DIR_PIN 6    // Direzione rotazione
#define ACC_PIN 2    // Interrupt accelerometro 
#define SENSOR_PIN 3 // Sensore IR feedback
#define ENABLE_PIN 4 // Attiva motore
#define HOLES 10     // Fori disco controllo 
#define STEPS_4_REV 400.0 // Passi per giro (modificabile con ponticelli on board)
#define DEBUG false
#define F_CLK   48000000UL

#define MPU_ADDR 0x68   //indirizzo accelerometro
#define READFREQ_HZ 200 //Frequenza lettura accelerometro
#define MAX_ACQUIRE_TIME 45  //massimo tempo di acquisizione sec.
#define MIN_ACQUIRE_TIME 10  //minimo tempo di acquisizione sec.


// --- Timer GPT ---
FspTimer stepTimer;        // genera PWM per STEP quando cambia velocità
FspTimer rampTimer;        // gestisce la rampa di frequenza
PwmOut pwmTimer(STEP_PIN); // genera PWM per STEP a velocità costante

const uint8_t SD_CS_PIN = 10;  //Pin scheda SD

const float MAX_ACC  =  3.00; //rad/sec^2
const float MIN_FREQ =  0.30; //Hz
const float MAX_FREQ = 16.00; //Hz
const int   rampIntervalMs = 50; // ms, intervallo di aggiornamento rampa (dt)
int j = 0, i = 0; //contatori
float targetFreq, currentFreq, freqRev = 0.0;
float countFreq[5] = {0};
int revolution = 0;
float Average = 0, Sum = 0;
volatile bool pinLevel = LOW;
uint32_t acquireTime = 0;

volatile float targetStepFreq, currentStepFreq, rampStepAcc;
volatile int hole_counter = 0; // --- PID (su RPM) ---
volatile unsigned long t_start = 0;
volatile unsigned long t_hole = 0;
unsigned long revPeriodMicros = 0;
unsigned int freq_prescaler = 0;
uint32_t sincroTime = 0;
uint8_t activeAddr = 0x68;

uint16_t bufIndex = 0;
volatile bool fifoFlag = false;  //diventa true quando IL FIFO è pieno durante logging
volatile bool loggerStop = false; //diventa true quando si da il comando LOGGER_OFF
bool firstPass = false;
volatile uint16_t count = 0;
volatile uint16_t TimeStampOffset = 0;
volatile uint32_t lastTimestamp = 0;
volatile uint16_t block = 0;
uint32_t startTime;
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

struct packet {
  uint16_t header = 0xAAAA;
  int16_t data[3];
  uint32_t timeStamp;
} __attribute__((packed));

packet dataBuf;
volatile uint32_t timeStamp;
volatile uint8_t activeBuff = 0;


// --- prototipi ---
void setupMPU(void);
void writeMPU(uint8_t reg, uint8_t val);
uint16_t readFIFOCount(void);
void onIrPulse(void);
void onStepTimer(timer_callback_args_t *args);
void onRampTimer(timer_callback_args_t *args);
void resetValue(void);
void handleCommand(char *cmd);
bool isFloat(const char *s);
void floatToString(float value, char *buf, size_t size, char sep);
int smartPrescaler(float freq);
int smartBegin(float freq);
void setupMPU(void);
inline void setBit(char *arr, int index);
inline void clearBit(char *arr, int index);
inline bool readBit(const char *arr, int index);
bool readData(int16_t *out);
void HelpCommand(void);

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

//******************************************************
//SETUP
//******************************************************

void setup() {
  Serial.begin(115200); while (!Serial); //valutare 230400
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

  Wire.begin();
  Wire.setClock(400000);
  setupMPU();


  Serial.println(F("Parser pronto (max 3 decimali)."));
  Serial.println(F("Comandi: +, ++, +++, -, --, ---, +n, -n, n"));
  Serial.println(F("GO, STOP, LOGGER_ON, Tnn, LIST, RESET, FREQ, HELP, ?"));
  Serial.println(F("================================================================================"));
}

//***********************************************************************
//Loop Principale
//***********************************************************************
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
    //Serial.println(F("busy"));
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
    Serial.print(F("Step acceleration: ")); Serial.print(rampStepAcc, 3); Serial.println(F(" step/sec^2"));
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
    Serial.println(F("Start"));
    delay(10); // Piccola pausa per far uscire il testo
    setupMPU();
    Serial.flush(); 
    // Letture a vuoto
    for (uint8_t i = 0; i < 5; i++) readData(dataBuf.data); //aumentare valore 5, se del caso
    // ACQUISIZIONE ---

    startTime = micros();
    sincroTime = t_start; //tempo dell'ultimo passaggio buco (opposto alla massa rotante)
    
    uint32_t startLoggedTime = millis();
    uint32_t nextSampleTime = micros();
    uint16_t readSamples = 0;
    // 3. Loop di acquisizione
    //Serial.print(acquireTime);
    while (millis() - startLoggedTime < acquireTime) {
      // Sincronizzazione rigida a 200Hz (5000 microsecondi)
      if (micros() >= nextSampleTime) {
        dataBuf.timeStamp = nextSampleTime; // Timestamp teorico per evitare drift
        if (readData(dataBuf.data)) {
          // Spedisce i 12 byte binari
          Serial.write((uint8_t *)&dataBuf, sizeof(dataBuf));
          readSamples++;
        }
        nextSampleTime += 5000; //5000 per 200 Hz
      }
    
      // Qui il processore può fare piccole operazioni di background 
      // purché durino meno di 1-2 ms
    }
    // --- Fine loop di acquisizione ---
    
    // Preparo l'ultimo record di chiusura (12 byte totali come gli altri)
    dataBuf.header = 0xAAAA;
    dataBuf.timeStamp = sincroTime;    // Tempo di sincronismo
    
    // Suddivido startTime (32 bit) in due int16_t
    dataBuf.data[0] = (int16_t)(startTime & 0xFFFF);        // Parte bassa
    dataBuf.data[1] = (int16_t)((startTime >> 16) & 0xFFFF); // Parte alta
    dataBuf.data[2] = (uint16_t)((micros() - startTime) / 1000); // Durata in ms (più utile dei sec); 

    // Spedisco l'ultimo pacchetto binario
    Serial.write((uint8_t *)&dataBuf, sizeof(dataBuf));

    // 2. Inviamo il record di chiusura (Sentinel): 12 byte tutti a 0xFF
    uint8_t sentinel[12];
    memset(sentinel, 0xFF, 12);
    Serial.write(sentinel, 12);
    
    Serial.flush(); // Aspetta che i bit escano fisicamente
    
    process.dataLogger = false;
    Serial.println(F("Data Logger disattivato."));
    Serial.print(F("Campionati n. ")); Serial.print(readSamples); Serial.print(F(" punti.")); //n. dati salvati
    Serial.print(F("Tempo di campionamento ")); Serial.print((float)(millis() - startLoggedTime) / 1000.0, 3); Serial.println(F(" sec."));
    block = 0;

    // Aspetta conferma da Python prima di tornare in modalità testuale
    bool confirmed = false;
    unsigned long startWait = millis();

    while (!confirmed && (millis() - startWait < 3000)) { // Timeout 3s per sicurezza
      if (Serial.available()) {
        String resp = Serial.readStringUntil('\n');
        if (resp.indexOf("WRITEOK") >= 0) {
            confirmed = true;
        }
      }
    }
    Serial.println(F("available"));
  }

}

//==============================================================================
//Parser comandi
//==============================================================================
void handleCommand(char *cmd) {
  if (process.dataLogger) {
    Serial.print(F("Data Logger attivo. Comando ")); Serial.print(cmd); Serial.println (F(" non valido o non attivo."));
    return; // esce subito, ignorando il resto
  }
  if (cmd[0] == '+' || cmd[0] == '-') {
    int8_t sign = (cmd[0] == '+') ? 1 : -1;
    uint8_t len = strlen(cmd);
    if (isdigit(cmd[1])) {
      // Gestisce "+5.5" o "-2"
      targetFreq += atof(cmd);
      process.freqChanged = true;
    } 
    else if (len <= 3) {
      // Gestisce "+", "++", "+++", "-", "--", "---"
      // Tabella pesi: len 1 -> 0.001, len 2 -> 0.010, len 3 -> 0.100
      float weights[] = {0.001f, 0.010f, 0.100f};
      targetFreq += sign * weights[len - 1];
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
  else if (cmd[0] == 'T' && isdigit(cmd[1])) {
    // 1. Leggi i secondi (es. 5)
    uint32_t sec = atol(cmd + 1); 
    
    // 2. Applica i limiti di sicurezza sui secondi
    sec = constrain(sec, MIN_ACQUIRE_TIME, MAX_ACQUIRE_TIME); 

    // 3. Aggiungi il secondo di bonus e converti in ms
    // Usiamo l'assegnazione diretta '=' invece di '+=' o '++'
    acquireTime = (sec + 1) * 1000; 

    Serial.print(F("Logger impostato per: ")); 
    Serial.print(sec); 
    Serial.println(F(" sec (+1 extra)."));
  }
  else if (strcmp(cmd, "LOGGER_ON")   == 0) {
    //Attivare il data logger
    process.dataLogger = true;
  }
  else if (strcmp(cmd, "A") == 0) {
    Serial.print(F("available"));
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
  else if (strcmp(cmd, "HELP")  == 0 || (strcmp(cmd, "?")  == 0)) { HelpCommand();
  }  else {
    Serial.print(F("Comando sconosciuto: ")); Serial.println(cmd);
  }

  if (process.freqChanged) {
    targetFreq = constrain(roundf(targetFreq * 1000.0f) / 1000.0, MIN_FREQ, MAX_FREQ); // arrotondamento a 3 decimali
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

//   LETTURA GENERICA
bool readData(int16_t *out) {
  Wire.beginTransmission(activeAddr);
  Wire.write(0x3B);  // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(activeAddr, (byte)6);
  if (Wire.available() == 6) {
    out[0] = (Wire.read() << 8) | Wire.read(); //X acc.
    out[1] = (Wire.read() << 8) | Wire.read(); //Y acc.
    out[2] = (Wire.read() << 8) | Wire.read(); //Z acc.
    return true;
  }
  return false;
}

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
}

//==============================================================================
//Fine gestione GY-521
//==============================================================================

//==============================================================================
// Converte float in string sostituendo il separatore
//==============================================================================
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
//Help command
//==============================================================================
void HelpCommand (void) {
  Serial.println(F("+/-     -> Aggiunge/toglie 0.001 Hz dalla frequenza corrente"));
  Serial.println(F("++/--   -> Aggiunge/toglie 0.010 Hz dalla frequenza corrente"));
  Serial.println(F("+++/--- -> Aggiunge/toglie 0.100 Hz dalla frequenza corrente"));
  Serial.println(F("+n/-n   -> Aggiunge/toglie n Hz dalla frequenza corrente"));
  Serial.println(F("n       -> Imposta ad n la frequenza corrente"));
  Serial.println(F("Tnn     -> imposta la drata del logger in sec. min 10, max 45"));
  Serial.println(F("GO/STOP -> Avvia/ferma il motore"));
  Serial.println(F("LIST    -> Stampa sul monitor seriale report di funzionamento"));
  Serial.println(F("RESET   -> Reset"));
  Serial.println(F("FREQ    -> Stampa sul monitor seriale la frequenza impostata e l'ultima rilevata"));
  Serial.println(F("?/HELP  -> Stampa sul monitor seriale l'elenco dei comandi"));
}


