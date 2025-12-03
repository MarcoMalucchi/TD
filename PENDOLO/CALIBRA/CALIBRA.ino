#include <Arduino.h>
#include <SdFat.h>
#include <Wire.h>

#define RAD_TO_DEG (180.0 / PI)
#define PRINT_ANGLE false
#define TYPE_DISTRIB 1 //1 angolo aureo; 2 spirale equidistante

const uint8_t SD_CS = 10; 

// ===============================
//  STRUTTURE
// ===============================
struct elab {
  float covariance[6];
  float average[3];
} __attribute__((packed));

const uint16_t Nsamples = 1850; // campioni per ciascun batch
const uint16_t stock = 20;      // campioni di riserva 
const uint16_t n = 150;         // numero massimo batch memorizzabili
const float k = 8;              // distanza dalla media x scarto campioni

elab Misure[n];
uint8_t batchIndex = 0;

// ===============================
//  MEMORIA PER ACQUISIZIONE
// ===============================
int16_t data[Nsamples+stock][3];

SdFat SD;
uint32_t t_start = 0;

// =====================================================
//   RICONOSCIMENTO AUTOMATICO ACCELEROMETRO
// =====================================================

enum SensorType {
  NONE = 0,
  S_LIS3DH,
  S_ADXL345,
  S_MPU6050
};

SensorType detectedSensor = NONE;
uint8_t activeAddr = 0;

// --- Funzioni base I2C generiche ---
bool probeI2C(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

uint8_t readRegRaw(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, (uint8_t)1);
  return (Wire.available() ? Wire.read() : 0xFF);
}

void writeRegRaw(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

// =====================================================
//   RILEVAMENTO SENSORI
// =====================================================
SensorType detectSensor() {

  // ----- LIS3DH (0x18/0x19 → ID = 0x33) -----
  if (probeI2C(0x18)) {
    if (readRegRaw(0x18, 0x0F) == 0x33) { activeAddr = 0x18; return S_LIS3DH; }
  }
  if (probeI2C(0x19)) {
    if (readRegRaw(0x19, 0x0F) == 0x33) { activeAddr = 0x19; return S_LIS3DH; }
  }

  // ----- ADXL345 (0x53 → DEVID = 0xE5) -----
  if (probeI2C(0x53)) {
    if (readRegRaw(0x53, 0x00) == 0xE5) { activeAddr = 0x53; return S_ADXL345; }
  }

  // ----- MPU6050 (0x68 → WHO_AM_I = 0x68) -----
  if (probeI2C(0x68)) {
    if (readRegRaw(0x68, 0x75) == 0x68) { activeAddr = 0x68; return S_MPU6050; }
  }

  return NONE;
}

// =====================================================
//   INIZIALIZZAZIONE SENSORI
// =====================================================

void init_LIS3DH() {
  writeRegRaw(activeAddr, 0x24, 0x80); // BOOT = 1
  delay(5);
  writeRegRaw(activeAddr, 0x20, 0x00); // Power-down
  delay(10);
  writeRegRaw(activeAddr, 0x24, 0x00); // CTRL_REG5: FIFO_EN = 0
  writeRegRaw(activeAddr, 0x2E, 0x00); // FIFO_CTRL_REG: FIFO mode = BYPASS
  writeRegRaw(activeAddr, 0x20, 0x67); // ODR 200Hz, XYZ enable
  writeRegRaw(activeAddr, 0x21, 0x00); // filtro disattivato
  writeRegRaw(activeAddr, 0x23, 0x08); // High Res, ±2g
  delay(5);
}

void init_ADXL345() {

  writeRegRaw(activeAddr, 0x2D, 0x00); // ADXL345 disable
  delay(10);
  writeRegRaw(activeAddr, 0x38, 0x00); // FIFO bypass (svuota buffer)
  delay(10);
  writeRegRaw(activeAddr, 0x31, 0x00); // ±2g
  writeRegRaw(activeAddr, 0x2C, 0x0A); // 200 Hz
  writeRegRaw(activeAddr, 0x2F, 0x00); // Nessun interrupt 
  writeRegRaw(activeAddr, 0x2E, 0x00); // Nessun interrupt
  writeRegRaw(activeAddr, 0x2D, 0x08); // Measure mode
  delay(10);
}

void init_MPU6050() {
  writeRegRaw(activeAddr, 0x6B, 0x80); //bit reset
  delay(100);
  writeRegRaw(activeAddr, 0x1A, 0x02); //Low Pass filter 184 Hz
  writeRegRaw(activeAddr, 0x19, 0x09); //Sample rate 200 Hz
  writeRegRaw(activeAddr, 0x1C, 0x00); // ±2g
  writeRegRaw(activeAddr, 0x6B, 0x00); // Start
  delay(10);
}

// =====================================================
//   LETTURA GENERICA
// =====================================================

void readData(int16_t *out) {

  switch(detectedSensor) {

    // ----- LIS3DH -----
    case S_LIS3DH:
      Wire.beginTransmission(activeAddr);
      Wire.write(0x28 | 0x80);     // OUT_X_L con auto-increment
      Wire.endTransmission(false);
      Wire.requestFrom(activeAddr, (uint8_t)6);
      if (Wire.available() == 6) {
        out[0] = (Wire.read() | (Wire.read() << 8));
        out[1] = (Wire.read() | (Wire.read() << 8));
        out[2] = (Wire.read() | (Wire.read() << 8));
      }
      break;

    // ----- ADXL345 -----
    case S_ADXL345:
      Wire.beginTransmission(activeAddr);
      Wire.write(0x32);
      Wire.endTransmission(false);
      Wire.requestFrom(activeAddr, (uint8_t)6);
      if (Wire.available() == 6) {
        out[0] = Wire.read() | (Wire.read() << 8);
        out[1] = Wire.read() | (Wire.read() << 8);
        out[2] = Wire.read() | (Wire.read() << 8);
      }
      break;

    // ----- MPU6050 -----
    case S_MPU6050:
      Wire.beginTransmission(activeAddr);
      Wire.write(0x3B);  // ACCEL_XOUT_H
      Wire.endTransmission(false);
      Wire.requestFrom(activeAddr, (uint8_t)6);
      if (Wire.available() == 6) {
        out[0] = (Wire.read() << 8) | Wire.read();
        out[1] = (Wire.read() << 8) | Wire.read();
        out[2] = (Wire.read() << 8) | Wire.read();
      }
      break;
  }
}

String getSensorDataName(SensorType s) {
  switch (s) {
    case S_LIS3DH:
      return "dataLIS3DH";
    case S_ADXL345:
      return "dataADXL345";
    case S_MPU6050:
      return "dataMPU6050";
    default:
      return "dataUNKNOWN";
  }
}

// ======================================
//  CALCOLO MEDIA CON FILTRO Mahalanobis (scarta se troppo distanti dalla media)
// ======================================
uint16_t filterAverage(int16_t mTot[][3], int16_t n, int16_t stock, float k, float Avg[3]) {
  uint16_t Ntot = n + stock;
  // ----- 1) MEDIA PRELIMINARE -----
  float avgP[3] = {0,0,0};
  for (uint16_t i = 0; i < Ntot; i++) {
    avgP[0] += mTot[i][0];
    avgP[1] += mTot[i][1];
    avgP[2] += mTot[i][2];
  }
  avgP[0] /= Ntot;
  avgP[1] /= Ntot;
  avgP[2] /= Ntot;
  // ----- 2) DEVIAZIONE STANDARD -----
  float std[3] = {0,0,0};
  for (uint16_t i = 0; i < Ntot; i++) {
    std[0] += sq(mTot[i][0] - avgP[0]);
    std[1] += sq(mTot[i][1] - avgP[1]);
    std[2] += sq(mTot[i][2] - avgP[2]);
  }
  std[0] = sqrt(std[0] / (Ntot - 1));
  std[1] = sqrt(std[1] / (Ntot - 1));
  std[2] = sqrt(std[2] / (Ntot - 1));

  // ----- 3) FILTRO + SWAP TRA OUTLIER E STOCK -----
  int idxStock = n;
  for (uint16_t i = 0; i < n; i++) {
    float dx = (mTot[i][0] - avgP[0]) / std[0];
    float dy = (mTot[i][1] - avgP[1]) / std[1];
    float dz = (mTot[i][2] - avgP[2]) / std[2];
    float d = sqrt(dx*dx + dy*dy + dz*dz);
    bool valido = (d <= k);
    if (!valido)
    {
      // --- cerca stock valido ---
      bool stockTrovato = false;
      while (idxStock < Ntot && !stockTrovato) {
        float dxs = (mTot[idxStock][0] - avgP[0]) / std[0];
        float dys = (mTot[idxStock][1] - avgP[1]) / std[1];
        float dzs = (mTot[idxStock][2] - avgP[2]) / std[2];
        float ds  = sqrt(dxs*dxs + dys*dys + dzs*dzs);
        if (ds <= k) {
          // ---- SWAP IN PLACE ----
          for (uint16_t c = 0; c < 3; c++) {
            float tmp         = mTot[i][c];
            mTot[i][c]        = mTot[idxStock][c];
            mTot[idxStock][c] = tmp;
          }
          stockTrovato = true;
        }
        idxStock++;  // passa allo stock successivo
      }
            // se nessuno stock è valido → manteniamo il dato originale (outlier)
    }
  }

  // ----- 4) MEDIA FINALE (su mTot filtrata) -----
  Avg[0] = Avg[1] = Avg[2] = 0;
  for (uint16_t i = 0; i < n; i++) {
    Avg[0] += mTot[i][0];
    Avg[1] += mTot[i][1];
    Avg[2] += mTot[i][2];
  }
  Avg[0] /= n;
  Avg[1] /= n;
  Avg[2] /= n;
  return (idxStock - n);
}

//=============================================================
//  CALCOLO ANGOLI
//=============================================================
void computeAnglesFromVector(const float Avg[3],
                             float &azimuth_deg,     // piano XY, rispetto asse X (0..360)
                             float &elevation_deg,   // elevazione rispetto piano XY (-90..90)
                             float &planeYZ_deg,     // piano YZ, rispetto asse Y (0..360)
                             float &planeXZ_deg,     // piano XZ, rispetto asse X (0..360)
                             float &angleX_deg,      // angolo tra vettore e asse X (0..180)
                             float &angleY_deg,      // angolo tra vettore e asse Y (0..180)
                             float &angleZ_deg)      // angolo tra vettore e asse Z (0..180)
{
  float x = Avg[0];
  float y = Avg[1];
  float z = Avg[2];

  // norma (modulo) del vettore
  float norm = sqrt(x*x + y*y + z*z);
  if (norm == 0.0f) {
    // vettore nullo: im1300 postiamo a zero o a NaN a piacere
    azimuth_deg = elevation_deg = planeYZ_deg = planeXZ_deg = 0.0f;
    angleX_deg = angleY_deg = angleZ_deg = 0.0f;
    return;
  }

  // --- angoli planari (proiezioni) ---
  // piano XY: angolo rispetto ad asse X (atan2(y,x))
  float az = atan2(y, x) * RAD_TO_DEG;         // -180 .. +180
  if (az < 0) az += 360.0f;                    // 0 .. 360
  azimuth_deg = az;

  // piano YZ: angolo della proiezione sul piano YZ rispetto asse Y
  float pyz = atan2(z, y) * RAD_TO_DEG;
  if (pyz < 0) pyz += 360.0f;
  planeYZ_deg = pyz;

  // piano XZ: angolo della proiezione sul piano XZ rispetto asse X
  float pxz = atan2(z, x) * RAD_TO_DEG;
  if (pxz < 0) pxz += 360.0f;
  planeXZ_deg = pxz;

  // --- azimut / elevazione (coordinate sferiche utili) ---
  // azimut = azimuth_deg (già calcolato)
  // elevazione = atan2(z, sqrt(x^2 + y^2)) -> -90 .. +90
  elevation_deg = atan2(z, sqrt(x*x + y*y)) * RAD_TO_DEG;

  // --- angoli traz, XYZ on)

  // High-resolution ± vettore e ciascun asse (usando prodotto scalare) ---
  // angle between vector and X-axis = acos(x / norm)
  float vx = x / norm;
  float vy = y / norm;
  float vz = z / norm;

  // protezione numerica per acos (valori devono stare in [-1,1])
  if (vx > 1.0f) vx = 1.0f; if (vx < -1.0f) vx = -1.0f;
  if (vy > 1.0f) vy = 1.0f; if (vy < -1.0f) vy = -1.0f;
  if (vz > 1.0f) vz = 1.0f; if (vz < -1.0f) vz = -1.0f;

  angleX_deg = acos(vx) * RAD_TO_DEG;   // 0..180
  angleY_deg = acos(vy) * RAD_TO_DEG;
  angleZ_deg = acos(vz) * RAD_TO_DEG;
}

// ===============================
//  FUNZIONE DI ACQUISIZIONE
// ===============================
void acquireAndProcess() {
  if (batchIndex >= n) {
      Serial.println("Memoria Misure[] piena, impossibile acquisire oltre.");
      return;
  }

  int16_t sample[3];
  Serial.print("Acquisizione batch #"); Serial.println(batchIndex);
  uint32_t t1 = micros();
  // --- 1) ACQUISIZIONE ---
  for (int i = 0; i < Nsamples+stock; i++) {
      readData(sample);
      data[i][0] = sample[0];
      data[i][1] = sample[1];
      data[i][2] = sample[2];
      delay(5);
  }
  float tl = (micros()-t1)/1000000.0;
  Serial.print(F("Tempo di lettura sensore: "));Serial.print(tl, 2);Serial.print(F(" sec.\t"));
  Serial.print(F("Frequenza di lettura sensore: "));Serial.print(Nsamples/tl, 0);Serial.print(F(" Hz.\n"));
  // --- 2) MEDIA ---
  float Avg[3] = {0};
    
  uint16_t refuse = filterAverage(data, Nsamples, stock, k, Avg);
  Serial.print(F("Scartati n. ")); Serial.print(refuse);Serial.println(F(" campioni."));
  // --- 3) COVARIANZA ---
  float cov[6] = {0};

  for (int i = 0; i < Nsamples; i++) {
    float dx[3];
    for (int j = 0; j < 3; j++)
      dx[j] = data[i][j] - Avg[j];

    for (int r = 0; r < 3; r++){
      uint8_t j;
      for (int c = r; c < 3; c++) {
        j = (r*2 - (r*(r-1))/2 + c);
        cov[j] += dx[r] * dx[c];
      }
    }
  }

  for (uint8_t j = 0; j < 6; j++)
    cov[j] /= (Nsamples - 1);

    // --- 4) SALVATAGGIO NELLA STRUTTURA ---
  for (uint8_t j = 0; j < 3; j++)
    Misure[batchIndex].average[j] = Avg[j];

  for (uint8_t j = 0; j < 6; j++)
    Misure[batchIndex].covariance[j] = cov[j];

  Serial.print("Batch #");uint32_t t_start = 0;
  Serial.print(batchIndex++);
  Serial.println(" acquisito ed elaborato.");
  stampaBatch(batchIndex-1);
}

// ===============================
//  SALVATAGGIO BINARIO
// ===============================
void salvaSuFile() {
  Serial.println("Salvataggio su SD...");
  String FileName = getSensorDataName(detectedSensor) + ".bin";
  FsFile f = SD.open(FileName, O_WRITE | O_CREAT | O_TRUNC);
  if (!f) {
    Serial.println("ERRORE: impossibile creare dati.bin");
    return;
  }
  // Salvi solo i batch realmente acquisiti
  f.write((uint8_t *)&batchIndex, sizeof(int));
  for (int i = 0; i < batchIndex; i++) {
    f.write((uint8_t *)&Misure[i], sizeof(elab));
  }
  f.close();
  Serial.print(batchIndex);Serial.println(" campioni salvati nel file dati.bin su SD.");
}

// ===============================
//  STAMPA DIAGNOSTICA
// ===============================
void stampaBatch(int k) {
  if (k >= batchIndex) return;

  Serial.print("\n=== Batch #");
  Serial.print(k);
  Serial.println(" ===");

  Serial.println(F("Media:"));
  Serial.println(("X             Y          Z"));
  for (int i = 0; i < 3; i++) {
    Serial.print(Misure[k].average[i], 6);
    Serial.print("  ");
  }
  Serial.println();
  Serial.println(F("Deviazione standard:"));
  for (int i = 0; i < 3; i++) {
    int j = 3*i-(i*(i-1)/2);
    Serial.print(sqrt(Misure[k].covariance[j]), 6);
    Serial.print("  ");
  }
  Serial.println();
  if (!PRINT_ANGLE) {
    Serial.println(F("\nAngoli:"));
    Serial.println(F("xy      xz      yz"));
    Serial.print(RAD_TO_DEG*atan(Misure[k].average[1]/Misure[k].average[0]));Serial.print(F("  "));
    Serial.print(RAD_TO_DEG*atan(Misure[k].average[0]/Misure[k].average[2]));Serial.print(F("  "));
    Serial.println(RAD_TO_DEG*atan(Misure[k].average[1]/Misure[k].average[2]));
  } else {
    float az, el, pyz, pxz, ax, ay, azz;
    computeAnglesFromVector(Misure[k].average, az, el, pyz, pxz, ax, ay, azz);

    Serial.print("\nAzimusth (XY): "); Serial.println(az);
    Serial.print("Elevation: "); Serial.println(el);
    Serial.print("Plane YZ angle (w.r.t Y): "); Serial.println(pyz);
    Serial.print("Plane XZ angle (w.r.t X): "); Serial.println(pxz);
    Serial.print("Angle with X axis: "); Serial.println(ax);
    Serial.print("Angle with Y axis: "); Serial.println(ay);
    Serial.print("Angle with Z axis: "); Serial.println(azz);    
  }

  Serial.println(F("\nCovarianza:"));
  for (int j = 0; j < 6; j++) {
    Serial.print(Misure[k].covariance[j], 6);
    Serial.print("\t");

  }
  Serial.println();
}

void make_point(uint16_t type) {
  switch (type) {
    case 1: {//angolo aureo
      float aureo = 180.0*(3.0-sqrt(5.0));
      for(int j=0; j < n; j++){
        Misure[j].average[0]=acos(1-(2*(float)j/(n-1)))*180/PI;
        Misure[j].average[1]=(j*aureo-int(j*aureo/360)*360);
      }
      break;
    }
    case 2:{ //spirale equidistante
      const float a =  3.99928688f;
      const float b =  0.04715093f;
      const float c =  2.21065379f;
      const float d = -1.70875050f;

      // supponiamo n, theta_arr[], phi_arr[] già esistano e siano globali
      float N = sqrtf(PI * (n - 1)) * 0.5f;

      // calcola S con il fit ottimizzato (N>~1 consigliato)
      float S = a * N + b + c / N + d / (N * N);

      // per ogni punto equispaziato s_k = (j/(n-1))*S
      for (int j = 0; j < n; ++j) {
        float s = S * ((float)j / (float)(n - 1));

        // inversione analitica approssimata:
        float v = 1.0f - 2.0f * (s / S);
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        float t = acosf(v) / PI;

        Misure[j].average[0] = 180.0f * t;
        Misure[j].average[1] = 360.0f * (N * t - int(N * t));
      }
      break;
    }
  }
}

// ===============================
//  SETUP
// ===============================
void setup() {
  Serial.begin(115200); while (!Serial);
  Wire.begin(); Wire.setClock(400000);
  delay(100);

  Serial.println("Riconoscimento accelerometro...");
  detectedSensor = detectSensor();

  if (detectedSensor == NONE) {
    Serial.println("ERRORE: nessun sensore I2C riconosciuto!");
    while(1);
  }
  switch(detectedSensor) {
    case S_LIS3DH:  Serial.println("Trovato LIS3DH");  init_LIS3DH();  break;
    case S_ADXL345: Serial.println("Trovato ADXL345"); init_ADXL345(); break;
    case S_MPU6050: Serial.println("Trovato MPU6050"); init_MPU6050(); break;
  }
   
  if (!SD.begin(SD_CS, SD_SCK_MHZ(25))) {
    Serial.println("Errore inizializzazione SD!");
    while (1);
  }
    
  make_point(TYPE_DISTRIB); 

  Serial.println("Pronto.");
  Serial.print(F("Premi 'a' per acquisire un batch di ")); Serial.print(Nsamples); Serial.println(F(" misure."));
  Serial.println(F("Premi 's' per salvare tutte le elaborazioni in dati.bin."));
  Serial.println(F("Premi 'r' per sovrascrivere l'ultima acquisizione"));
  Serial.print("Altezza iniziale: ");Serial.print(Misure[0].average[0],0);
  Serial.print(",  Azimut iniziale: ");Serial.println(Misure[0].average[1],0);
}

// ===============================
//  LOOP
// ===============================
void loop() {
  uint32_t ti, tf;
  if (Serial.available()) {
    char c = Serial.read();
    while (Serial.available() > 0) Serial.read();

    if (c == 'a') {
      if (t_start == 0) t_start = millis();
      ti=micros();
      acquireAndProcess();
      tf = micros();
    }
    else if (c == 's') salvaSuFile();
    else if (c == 'r') {
      batchIndex == 0 ? 0 : batchIndex--;
      Serial.print(F("Riscrittura del campione n. ")); Serial.println(batchIndex);
    }
    else {Serial.print(F("Comando non valido:"));Serial.print(c, HEX);}
    Serial.print(F("Tempo acquisizione: "));Serial.print((float)(tf-ti)/1000000,1);Serial.print(F(" sec.\t"));

    int t_left = ((millis()-t_start)/batchIndex)*(n-batchIndex)/1000.0;
    Serial.print(F("Tempo rimanente:"));
    Serial.print(t_left/3600);Serial.print(":");Serial.print((t_left % 3600) / 60);Serial.print(":");Serial.println(t_left % 60);

    Serial.println("In attesa di nuovo 'a' o 's' o 'r' ...");
    Serial.print("Altezza: ");Serial.print(Misure[batchIndex].average[0],0);
    Serial.print(",  Azimut: ");Serial.println(Misure[batchIndex].average[1],0);
  }
}

