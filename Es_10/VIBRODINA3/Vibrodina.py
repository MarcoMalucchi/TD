import serial
import time
import threading
import sys
import os
import re
from datetime import datetime

# ==== CONFIGURAZIONE ====
PORT = '/dev/ttyACM0'
BAUD = 115200       #eventualmente raddoppiabile a 230400
PARAMETERS = ["+0.1"]     # incremento in frequenza (positivo o negativo) o frequenze (freq max 11.5Hz, freq min 0.3Hz)
INIT = 11.6      # frequenza iniziale in Hz (parte proprio da qui)
REPEAT_N = 42    # numero di passi in freq, ignorato assieme a  INIT se si specificano le frrequenze in PARAMETERS,
                # oppure numero di acquisizioni in modalità manuale (OPERATING_MODE = 0)
LOG_DURATION = 30  # durata acquisizione in secondi
OPERATING_MODE = 1  # 0 = manuale, 1 = automatica (usa PARAMETERS e REPEAT_N)
BASE_PATH = os.path.expanduser("/home/marco/Desktop/Uni_anno3/TD/Es_10/acquisizioni/parte_1/spazzata_completa/")
# =========================

# Variabili di Stato Globali
arduino_state = "unknown"
stop_flag = False
numFile = 0
sessionNum = 0  # Identificativo sessione (Snn)
fileName = ""
lock = threading.Lock()
session_log = []

def serial_listener(ser):
    global arduino_state, stop_flag, fileName
    while not stop_flag:
        try:
            if ser.in_waiting > 0:
                line_raw = ser.readline()
                line = line_raw.decode('utf-8', errors='ignore').strip()

                if line == "Start":
                    with lock:
                        target_file = fileName
                    processa_binario_raw(ser, target_file)

                elif line:
                    print(f"Arduino: {line}")
                    low = line.lower()
                    if low in ["busy", "available"]:
                        with lock:
                            arduino_state = low
        except Exception as e:
            if not stop_flag:
                print(f"\n[Errore Listener]: {e}")
        time.sleep(0.01)

def processa_binario_raw(ser, target_file):
    sentinel = b'\xff' * 5
    print(f"\n>>> Scrittura in corso: {os.path.basename(target_file)}")

    try:
        with open(target_file, "wb") as f:
            while not stop_flag:
                if ser.in_waiting > 0:
                    chunk = ser.read(ser.in_waiting)
                    if sentinel in chunk:
                        pos = chunk.find(sentinel)
                        f.write(chunk[:pos])
                        break
                    else:
                        f.write(chunk)
                time.sleep(0.001)

        ser.write(b"WRITEOK\n")
        print(f">>> File salvato e confermato.")
    except Exception as e:
        print(f"Errore durante il salvataggio: {e}")

def wait_for_state(state):
    global arduino_state
    print(f"⏳ Attendo stato '{state}'...")
    while not stop_flag:
        with lock:
            if arduino_state == state:
                print(f"✅ Arduino è '{state}'.")
                if state == "available":
                    arduino_state = "busy"
                return
        time.sleep(0.05)

def set_arduino_state(new_state):
    global arduino_state
    with lock:
        arduino_state = new_state.lower()

def send_command(ser, cmd):
    ser.write((str(cmd) + "\n").encode())

def mostra_progresso_log(duration_sec, steps=20):
    delay = duration_sec / steps
    print(f"\n>>> Acquisizione in corso ({duration_sec}s)...")
    for i in range(steps + 1):
        percent = (i / steps) * 100
        bar = '█' * i + '-' * (steps - i)
        sys.stdout.write(f'\rProgress: |{bar}| {percent:.0f}%')
        sys.stdout.flush()
        if i < steps:
            time.sleep(delay)
    print("\n>>> Sincronizzazione file...")

def InitSessionAndCounter():
    """Determina il numero della nuova sessione e resetta il contatore file."""
    global numFile, sessionNum
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    # Cerchiamo il numero di sessione più alto esistente (Snn)
    pattern_session = re.compile(r"^S(\d{2})DATA")
    max_s = 0

    for filename in os.listdir(BASE_PATH):
        match = pattern_session.match(filename)
        if match:
            s_val = int(match.group(1))
            if s_val > max_s:
                max_s = s_val

    sessionNum = max_s + 1
    numFile = 0 # Ogni sessione riparte da DATA0000 (o puoi scegliere di continuare)

    print(f">>> NUOVA SESSIONE AVVIATA: S{sessionNum:02d}")
    print(f">>> I file verranno salvati in: {BASE_PATH}")

def getNextFileName(frequenza_hz):
    """Genera il nome file includendo il prefisso sessione: SnnDATA0000_Hz00_000.bin"""
    global numFile, sessionNum
    f_arr = round(frequenza_hz, 3)
    p_int = int(f_arr)
    p_dec = int(round((f_arr - p_int) * 1000))

    # Nuovo formato: SnnDATAnnnn_Hznn_nnn.bin
    name = f"S{sessionNum:02d}DATA{numFile:04d}_Hz{p_int:02d}_{p_dec:03d}.bin"
    numFile += 1
    return os.path.join(BASE_PATH, name)

def GenerateRamp(Start, n, increment, down=0.3, up=15.7):
    ramp = []
    for i in range(n + 1):
        val = Start + (i * increment)
        val = max(down, min(up, val))
        f_str = f"{val:.3f}"
        if f_str not in ramp:
            ramp.append(f_str)
        if val == up or val == down:
            break
    return ramp

def scrivi_riepilogo_sessione():
    """Genera il file di log della sessione includendo Snn nel nome."""
    global sessionNum
    if not session_log: return

    log_name = os.path.join(BASE_PATH, f"S{sessionNum:02d}_SESSION_LOG.txt")

    with open(log_name, "w") as f:
        f.write(f"RIEPILOGO SESSIONE S{sessionNum:02d}\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Modalità: {'Automatica' if OPERATING_MODE != 0 else 'Manuale'}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'NOME FILE':<35} | {'PARAMETRO (Hz)':<15}\n")
        f.write("-" * 60 + "\n")
        for file, param in session_log:
            f.write(f"{file:<35} | {param:<15}\n")

    print(f"\n>>> Riepilogo sessione salvato: {os.path.basename(log_name)}")

def main():
    global stop_flag, fileName, session_log
    ser = None
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)
        print(f"Connesso a {PORT}")

        # Inizializza sessione e contatori
        InitSessionAndCounter()

        listener = threading.Thread(target=serial_listener, args=(ser,), daemon=True)
        listener.start()

        send_command(ser, "STOP")
        send_command(ser, "T" + str(LOG_DURATION))
        set_arduino_state("busy")

        if OPERATING_MODE != 0:
            if len(PARAMETERS) == 1 and REPEAT_N > 0:
                cmds = GenerateRamp(INIT, REPEAT_N, float(PARAMETERS[0]))
            else:
                cmds = PARAMETERS
            send_command(ser, "GO")
            wait_for_state("available")
        else:
            print("\n>>> MODALITÀ MANUALE ATTIVA")
            cmds = ["0.000"] * (REPEAT_N if REPEAT_N > 0 else 1)

        for j, param in enumerate(cmds):
            print(f"\n--- TEST {j+1}/{len(cmds)} ({param} Hz) ---")

            if OPERATING_MODE != 0:
                send_command(ser, param)
                time.sleep(20)      # Attendi 10 secondi per stabilizzazione oscillazione sistema prima di iniziare misura
                f_val = float(param)
                wait_for_state("available")
            else:
                input("Sistemare sensore e premere INVIO...")
                f_val = 0.0

            prossimo_path = getNextFileName(f_val)
            with lock:
                fileName = prossimo_path

            session_log.append((os.path.basename(prossimo_path), param))

            send_command(ser, "LOGGER_ON")
            mostra_progresso_log(LOG_DURATION)
            wait_for_state("available")
            print(f"✅ Completato.")
            time.sleep(5)

        scrivi_riepilogo_sessione()

    except KeyboardInterrupt:
        
        print("\nInterrotto.")
    except Exception as e:
        print(f"\nErrore: {e}")
    finally:
        if ser and ser.is_open:
            send_command(ser, "STOP")
            stop_flag = True
            time.sleep(0.5)
            ser.close()
            print("Sessione chiusa.")

if __name__ == "__main__":
    main()

