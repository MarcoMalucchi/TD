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

#define BUF_LEN 1536     //Dimensione buffer - // multiplo di 24 e 1536*10 multiplo di 512
#define BLOCK_LEN 24     //Dati FIFO
#define READFREQ_HZ 400  //Frequenza campionamento accelerometro


// Registri principali
#define ADXL345_ADDR    0x53  //indirizzo accelerometro
#define REG_POWER_CTL   0x2D  
#define REG_DATA_FORMAT 0x31  //Fondo scala
#define REG_BW_RATE     0x2C  //Frequenza lettura
#define REG_FIFO_CTL    0x38  //FIFO
#define REG_DATAX0      0x32  //Registri dati
#define REG_FIFO_STATUS 0x39

#define GRAVITY_ACC     9.805  //Accelerazione di gravità

// --- Timer GPT --- 
FspTimer stepTimer;        // genera PWM per STEP quando cambia velocità
FspTimer rampTimer;        // gestisce la rampa di frequenza
PwmOut pwmTimer(STEP_PIN); // genera PWM per STEP a velocità costante


const uint8_t SD_CS_PIN = 10;  //Pin scheda SD
uint8_t Rate;

const float MAX_ACC  =  3.00; //rad/sec^2 
const float MIN_FREQ =  0.30; //Hz 
const float MAX_FREQ = 12.00; //Hz
const int   rampIntervalMs = 50; // ms, intervallo di aggiornamento rampa (dt) 
int j=0, i=0; //contatori
float targetFreq, currentFreq, freqRev = 0.0;
float countFreq[5] = {0};
int revolution = 0;
float Average=0, Sum=0;
volatile bool pinLevel = LOW;

volatile float targetStepFreq, currentStepFreq, rampStepAcc; 
volatile int hole_counter = 0; // --- PID (su RPM) --- 
volatile unsigned long t_start = 0; 
volatile unsigned long t_hole = 0; 
unsigned long revPeriodMicros = 0;
unsigned int freq_prescaler = 0;
uint32_t sincroTime = 0;

uint16_t bufIndex = 0;
volatile bool fifoFlag = false;
bool firstPass = false;
volatile uint16_t count = 0;
volatile uint16_t TimeStampOffset = 0;
volatile uint32_t lastTimestamp = 0;
volatile uint16_t block = 0;
uint32_t startLoggedTime;

struct system {
  unsigned int motorOn: 1; 
  unsigned int motorDir: 1; 
  unsigned int dataLogger: 1; 
  unsigned int freqChanged: 1; 
  unsigned int motorChanged:1;
  unsigned int Acc: 1;
  unsigned int feedback:1;
  unsigned int gateway: 1;
} process; 

const size_t CMD_SIZE = 16; 
char cmd[CMD_SIZE]; 
byte idx = 0; 

struct packet {
  uint8_t x_high;
  uint8_t x_low;
  uint8_t y_high;
  uint8_t y_low;
  uint8_t z_high;
  uint8_t z_low;
};

struct Sample {
  uint32_t timeStamp;
  struct packet dataRaw;
} __attribute__((packed));

Sample dataBuf[BUF_LEN];
volatile uint32_t timeStamp[2][BLOCK_LEN];
volatile uint8_t activeBuff = 0;

// Per SD
SdFat sd;
FsFile dataFile;

// --- prototipi ---
bool initSD(uint8_t csPin, uint32_t clockMHz = 25); //clockMHz predefinito 25
void resetADXL();
void writeReg(uint8_t reg, uint8_t val);
uint8_t readReg(uint8_t reg);
uint8_t readFIFOCount();
int smartBegin(float freq);
void resetValue();
void handleCommand(char *cmd);
bool isFloat(const char *s);
int findNextDataFile(const char *folder, char *outName, size_t outSize, float freq);
void floatToString(float value, char *buf, size_t size, char sep);
uint8_t RateFromFreq(float freqHz);
float FreqFromRate(uint8_t rate);


// reset variabili 
void resetValue() { 
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
  Average=0;
} 

// --- ISR sensore IR --- 
void onIrPulse() { 
  t_hole = micros();
  process.gateway = true;
} 

// --- ISR accelerometro --- 
void dataReadyAcc() { 
  lastTimestamp = micros();
  fifoFlag = true;
  //isrCount++;
}

// --- Rampa: applica incremento o decremento rispettando alpha_acc/alpha_dec --- 
void onRampTimer(timer_callback_args_t *args) { 
  if (!process.motorOn) return; //Se stepTimer non è attivo 
  currentStepFreq += rampStepAcc;
  if ((currentStepFreq - targetStepFreq) * rampStepAcc >= 0.0) currentStepFreq = targetStepFreq;
  stepTimer.set_frequency(currentStepFreq*2.0);
}

// --- Rampa: applica incremento o decremento rispettando alpha_acc/alpha_dec --- 
void onStepTimer(timer_callback_args_t *args) { 
  pinLevel = !pinLevel;
  digitalWrite(STEP_PIN, pinLevel);
} 

void setup() {
  Serial.begin(115200); while (!Serial); 
  pinMode(STEP_PIN,   OUTPUT); 
  pinMode(DIR_PIN,    OUTPUT); 
  pinMode(SENSOR_PIN, INPUT); // sensore IR 
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
  
  Wire.begin();
  Wire.setClock(400000);
  Rate = RateFromFreq(READFREQ_HZ);
  Serial.print(F("Frequenza di campionamento: "));Serial.print(FreqFromRate(Rate), 3);Serial.println(F(" Hz"));
  resetADXL();
   
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

  if (!process.dataLogger) {
    if (process.motorChanged) { 
      process.motorChanged = false; 
      if (!process.motorOn) { 
        resetValue(); 
      }
      else { 
        Serial.println("Motore START"); 
        digitalWrite(ENABLE_PIN, LOW); //Abilita il motore 
        stepTimer.start();
        process.freqChanged = true; 
      }
    }
    
    if (!process.motorOn) return; 

    if (process.freqChanged) {
      Serial.println(F("busy"));
      detachInterrupt(digitalPinToInterrupt(SENSOR_PIN)); //disabilita il controllo feedback
      hole_counter = 0;
      Sum = 0; Average = 0, revolution = 0; t_start = 0;
      memset(countFreq, 0, sizeof(countFreq));
      float acc = (targetFreq > currentFreq ? MAX_ACC : -MAX_ACC); //Calcolo accelerazione 
      targetStepFreq = targetFreq*STEPS_4_REV;
      rampStepAcc = acc*STEPS_4_REV*rampIntervalMs/2000/PI; 
      process.freqChanged = false;
      process.Acc = true;
      process.feedback = false;
      
      // Prepara il timer PWM
      pwmTimer.end(); //Attenzione, forse non serve
	  
      R_PFS->PORT[4].PIN[5].PmnPFS_b.PMR = 0; // Disattiva funzione periferica (PWM, UART, ecc.)
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
      targetTick = F_CLK/freq_prescaler/targetStepFreq;
      process.Acc = false;
      float realAcc = 0.0;
      currentFreq = targetFreq;
      float Dt = (endAcc-startAcc)/1000.0;
      realAcc = 2*PI*(currentFreq-inFreq)/Dt;

      Serial.print(F("Step frequency:    ")); Serial.print(targetStepFreq, 3); Serial.println(F(" Hz")); 
      Serial.print(F("Real acceleration: ")); Serial.print(realAcc,        3); Serial.println(F(" rad/sec^2")); 
      Serial.print(F("Time acceleration: ")); Serial.print(Dt); Serial.println(F(" sec"));
      if (DEBUG){
        Serial.print(F("freq_prescaler   ")); Serial.println(freq_prescaler);
      }

      attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), onIrPulse, RISING); //abilita il controllo feedback
    }

    if (process.gateway) {
      process.gateway = false;
      static uint16_t turn = 0;
      if (t_start == 0) {
        t_start = t_hole;
      } else {
        freqRev = round(1.0e9 / (float)(t_hole - t_start))/1000.0;
        t_start = t_hole;
        revolution++;
        
        // Calcola media mobile s 5 giri
        if (revolution > 1) {
          if (abs((freqRev-Average)*100/Average) > 5.0) freqRev = Average;
        } else Average = freqRev;
        Sum -= countFreq[revolution%5];
        Sum += freqRev;
        Average = round(1000*Sum / (revolution < 5 ? revolution : 5))/1000;
        countFreq[revolution%5]=freqRev;

        if (targetFreq-Average == 0) {
          if (turn/targetFreq > 2 && process.feedback){  
            Serial.print(F("Frequenza target: ")); Serial.print(targetFreq, 3);Serial.print(F(" Hz; "));
            Serial.print(F("Frequenza media su ")); Serial.print((revolution < 5 ? revolution : 5)); Serial.print(F(" giri: "));
            Serial.print(Average, 3);Serial.print(F(" Hz; "));
            Serial.print(F("raggiunta dopo: ")); Serial.print(revolution);Serial.println(F(" giri"));
            process.feedback = false;
            Serial.println(F("available"));
          } else turn++;
        } else {
          if (!process.feedback){
            turn=0;
            Serial.println(F("busy"));
            Serial.print(F("Perso sincronismo al giro: ")); Serial.print(revolution);
            Serial.print(F("; Frequenza misurata: ")); Serial.print(freqRev, 3);Serial.println(F(" Hz; "));
          }
          process.feedback = true;
        }
      }
    }
  } else {
    if (fifoFlag) {
      fifoFlag = false;
      if (DEBUG) {
        uint8_t dataFIFO = readFIFOCount();
        Serial.println(dataFIFO);
      }
      for (int i = 0; i <24; i++) { 
        Wire.beginTransmission(ADXL345_ADDR);
        Wire.write(0x32); // indirizzo DATAX0
        Wire.endTransmission(false);
        Wire.requestFrom(ADXL345_ADDR, (byte)6);
        dataBuf[bufIndex].dataRaw.x_low = Wire.read(); dataBuf[bufIndex].dataRaw.x_high = Wire.read();
        dataBuf[bufIndex].dataRaw.y_low = Wire.read(); dataBuf[bufIndex].dataRaw.y_high = Wire.read();
        dataBuf[bufIndex].dataRaw.z_low = Wire.read(); dataBuf[bufIndex].dataRaw.z_high = Wire.read();
        bufIndex++;
      }
      dataBuf[bufIndex-1].timeStamp = lastTimestamp;
      
      if (bufIndex >= BUF_LEN) {
        // Scrittura buffer
        dataFile.write((uint8_t*)dataBuf, sizeof(dataBuf));
        dataFile.flush();
        block++;
        bufIndex = 0;
      }
      if (DEBUG) {Serial.print(bufIndex);Serial.print(F(" ")); Serial.println(block);}
    }
  }
}

//==============================================================================
//Parser comandi
//==============================================================================
void handleCommand(char *cmd) {
  if (process.dataLogger) {
    if (strcmp(cmd, "LOGGER_OFF") == 0) {
      // qui dovrei svuotare FIFO
      uint8_t extraData = readFIFOCount()/6;
      if (extraData < 30) {
        for (uint8_t i = 0; i < extraData; i++) {
          Wire.beginTransmission(ADXL345_ADDR);
          Wire.write(0x32);
          Wire.endTransmission(false);
          Wire.requestFrom(ADXL345_ADDR, 6);
          
          dataBuf[bufIndex].dataRaw.x_low = Wire.read(); dataBuf[bufIndex].dataRaw.x_high = Wire.read();
          dataBuf[bufIndex].dataRaw.y_low = Wire.read(); dataBuf[bufIndex].dataRaw.y_high = Wire.read();
          dataBuf[bufIndex].dataRaw.z_low = Wire.read(); dataBuf[bufIndex].dataRaw.z_high = Wire.read();

          bufIndex++;
        }
      }
      //writeReg(REG_POWER_CTL, 0x00); // ADXL345 standby;
      resetADXL();
      fifoFlag = false;
      process.dataLogger = false;
      detachInterrupt(digitalPinToInterrupt(ACC_PIN)); //Disabilita interrupt
	  
      //sovrascrive l'ultimo record con sincronismo
      bufIndex--;
      dataBuf[bufIndex].timeStamp = sincroTime;
      dataBuf[bufIndex].dataRaw.x_low = 0;dataBuf[bufIndex].dataRaw.x_high = 0;
      dataBuf[bufIndex].dataRaw.y_low = 0;dataBuf[bufIndex].dataRaw.y_high = 0;
      dataBuf[bufIndex].dataRaw.z_low = 0;dataBuf[bufIndex].dataRaw.z_high = 0;
      bufIndex++;
      size_t bytesToWrite = (size_t)bufIndex * sizeof(Sample);
      if (DEBUG) {
        Serial.println(bufIndex);
        Serial.println(sizeof(Sample));
        Serial.println(bytesToWrite);
      }
      if (bytesToWrite) {
        dataFile.write((uint8_t*)dataBuf, bytesToWrite); //scrive quanto resta in ram con sincroTime come ultimo record
      }
      char availableFile[64];
      dataFile.getName(availableFile, size_t(availableFile));
      
      dataFile.close(); //chiude file
      Serial.println(F("Data Logger disattivato."));
      Serial.print(F("Campionati n. ")); Serial.print(block*BUF_LEN+bufIndex-1); Serial.print(F(" punti."));   //n. dati salvati
      Serial.print(F("nel file: ")); Serial.println(availableFile);
      Serial.print(F("Tempo di campionamento ")); Serial.print((micros()-startLoggedTime)/1000000.0,3);Serial.println(F(" sec."));
      Serial.println(F("available"));
      block = 0; bufIndex=0;
    } else {
      Serial.print(F("Data Logger attivo. Comando ")); Serial.print(cmd); Serial.println ((" non valido o non attivo."));
    }
    return; // esce subito, ignorando il resto
  }
  process.freqChanged = false; 
  if      (strcmp(cmd, "+++") == 0) { targetFreq += 0.100; process.freqChanged = true; }
  else if (strcmp(cmd, "++")  == 0) { targetFreq += 0.010; process.freqChanged = true; } 
  else if (strcmp(cmd, "+")   == 0) { targetFreq += 0.001; process.freqChanged = true; } 
  else if (strcmp(cmd, "---") == 0) { targetFreq -= 0.100; process.freqChanged = true; } 
  else if (strcmp(cmd, "--")  == 0) { targetFreq -= 0.010; process.freqChanged = true; } 
  else if (strcmp(cmd, "-")   == 0) { targetFreq -= 0.001; process.freqChanged = true; } 
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
  else if (strcmp(cmd, "GO")   == 0) { process.motorOn = true; process.motorChanged = true;} 
  else if (strcmp(cmd, "STOP") == 0) { process.motorOn = false; process.motorChanged = true;} 
  else if (strcmp(cmd, "LOGGER_ON")   == 0) { 
    if (process.motorOn) Serial.println("Logger attivo con motore spento");
    char availableFile[64];
    if (DEBUG) Serial.println(availableFile);
    if (findNextDataFile("DATA", availableFile, sizeof(availableFile), currentFreq) == 0) {
      if (DEBUG) Serial.println(availableFile);
      //Attivare il data logger
      process.dataLogger = true;
      memset(dataBuf, 0, sizeof(dataBuf));
      memset((void*)timeStamp, 0, sizeof(timeStamp));
      Serial.println(F("Logger attivo"));Serial.println(F("busy"));
      //Apro il file 
      dataFile = sd.open(availableFile, O_WRITE | O_CREAT);
      // Interrupt accelerometro 
      attachInterrupt(digitalPinToInterrupt(ACC_PIN), dataReadyAcc, RISING);
      writeReg(REG_POWER_CTL, 0x08); // ADXL345 enable
      startLoggedTime=micros();
      sincroTime = t_start;
  
    } else {
      Serial.println("Impossibile attivare data logger. Controlla SD"); //volendo si può valutare errore
    }
  }
  else if (strcmp(cmd, "OFF")  == 0) { process.dataLogger = false; } 
  else if (strcmp(cmd, "RESET") == 0) { resetValue(); }
  else if (strcmp(cmd, "LIST") == 0) { 
    Serial.print(F("Fequenza impostata: ")); Serial.println(currentFreq, 3); 
    Serial.print(F("Fequenza media su 5 giri: ")); Serial.println(Average, 3); 
    Serial.print(F("Motore: ")); Serial.println(process.motorOn ? F("ON") : F("OFF")); 
    Serial.print(F("Data Logger: ")); Serial.println(process.dataLogger ? F("ATTIVO") : F("FERMO")); 
    Serial.print(F("Frequenza campionamento: ")); Serial.print(FreqFromRate(Rate), 3);Serial.println(F(" Hz"));
  } 
  else if (strcmp(cmd, "FREQ") == 0) { 
    Serial.print(F("Fequenza impostata: ")); Serial.println(currentFreq, 3);
    Serial.print(F("Fequenza misurata media su 5 giri: ")); Serial.println(Average, 3); 
  }
	//else if (strcmp(cmd, "HELP")  == 0 || (strcmp(cmd, "?")  == 0)) { HelpCommand(); } 
	else if (strcmp(cmd, "PRINT")  == 0) { printLastDetection(); } 
  else { 
    Serial.print(F("Comando sconosciuto: ")); Serial.println(cmd);
  } 
  
  if (process.freqChanged) { 
    targetFreq = constrain(roundf(targetFreq * 1000.0f) / 1000.0, 0.3, 12.0); // arrotondamento a 3 decimali 
    Serial.print(F("Target frequency: ")); Serial.print(targetFreq, 3);Serial.println(F(" Hz"));
  }
}

// verifica se una stringa è un float valido 

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
  unsigned int tick = 48000000UL/prescaler/freq;
  pwmTimer.period_raw(tick);
  pwmTimer.pulseWidth_raw(tick/2);
  return prescaler;
}

int smartPrescaler(float freq) {
  if (freq < 280.0) return 64; 
  else if (freq < 1200.0) return 4;
  else return 1;

}

//==============================================================================
//Gestione ADXL345
//==============================================================================

// Legge il valore dal registro
uint8_t readReg(uint8_t reg){
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, (byte)1);
  return Wire.read();
}

// Scrive un valore nel registro
void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(false);
}

// Setup ADXL345
void resetADXL() {
  writeReg(0x2D, 0x00); // standby (Measure=0)
  delay(10);
  writeReg(0x38, 0x00); // FIFO bypass (svuota buffer)
  delay(10);
  
  // configurazioni di base
  writeReg(0x38, 0x98); // stream mode, watermark 24
  writeReg(0x31, 0x00); // ±2g, full_res=0
  writeReg(0x2C, Rate); // frequenza
  
  writeReg(0x2F, 0x02); // abilita interrupt 2
  writeReg(0x2E, 0x02); // abilita DATA_READY interrupt
  delay(10);
}

// FIFO count
uint8_t readFIFOCount() {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(REG_FIFO_STATUS);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, (uint8_t)1);
  return Wire.read();
}
//==============================================================================
//Fine gestione ADXL345
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

        // Se sono proprio 3 cifre valide
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
    if (buf[i] == '.') { buf[i] = sep; break; }
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

  float scale = GRAVITY_ACC/256; //fattore di conversione per dati raw accelerometro fondo scala 2g
  Sample s;
  uint32_t count = 0;

  // Leggi e stampa tutti i record
  while (file.read(&s, sizeof(Sample)) == sizeof(Sample)) {
    int16_t x = ((int16_t)s.dataRaw.x_high << 8) | s.dataRaw.x_low;
    int16_t y = ((int16_t)s.dataRaw.y_high << 8) | s.dataRaw.y_low;
    int16_t z = ((int16_t)s.dataRaw.z_high << 8) | s.dataRaw.z_low;
    char buffer[15];
    sprintf(buffer, "% 5d", count++);
    Serial.print("Record ");Serial.print(buffer);
    sprintf(buffer, "% 11d", s.timeStamp);
    Serial.print("; t=");Serial.print(buffer);
    dtostrf(x*scale, 8, 4, buffer); Serial.print(";  X="); Serial.print(buffer);
    dtostrf(y*scale, 8, 4, buffer); Serial.print(";  Y="); Serial.print(buffer);
    dtostrf(z*scale, 8, 4, buffer); Serial.print(";  Z="); Serial.println(buffer);
  }

  Serial.print("Totale record letti: ");
  Serial.println(count);

  file.close();
}

uint8_t RateFromFreq(float freqHz) {
  double offset = 2*(log(5)/log(2) - 4); // calcolo offset esatto
  return (uint8_t)round(log(freqHz)/log(2) - offset);
}

float FreqFromRate(uint8_t rate) {
  double offset = 2*(log(5)/log(2) - 4);
  return exp(log(2) * (rate + offset));
}
