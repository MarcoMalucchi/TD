import serial
import time
import threading
from datetime import datetime

# ==== CONFIGURAZIONE ====
PORT = '/dev/ttyACM0' 
BAUD = 115200
PARAMETERS = ["+0.25"]    
INIT = 1.5                
REPEAT_N = 2              
LOG_DURATION = 5
# =========================

arduino_state = "unknown"
stop_flag = False
lock = threading.Lock() 

def serial_listener(ser):
    global arduino_state, stop_flag
    while not stop_flag:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    print(f"ARDUINO: {line}") # Helpful prefix
                    low = line.lower()
                    # We look for keywords within the line
                    if "busy" in low:
                        with lock: arduino_state = "busy"
                    elif "available" in low:
                        with lock: arduino_state = "available"
        except Exception as e:
            print(f"Read Error: {e}")
        time.sleep(0.01)

def wait_for_state(state, timeout=10):
    """Attende lo stato con un timeout di sicurezza per evitare blocchi infiniti."""
    print(f"⏳ Attendo stato '{state}'...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        with lock:
            if arduino_state == state:
                print(f"✅ Arduino è ora '{state}'.")
                return True
        if stop_flag: return False
        time.sleep(0.05)
    print(f"❌ TIMEOUT: Arduino non è entrato in stato '{state}'")
    return False

def send_command(ser, cmd):
    global arduino_state
    with lock:
        arduino_state = "unknown" # RESET state before sending
    ser.reset_input_buffer()      # Clear old messages
    print(f"▶️ Invio comando: {cmd}")
    ser.write((cmd + "\n").encode())

def main():
    global stop_flag

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2) 
        
        listener = threading.Thread(target=serial_listener, args=(ser,), daemon=True)
        listener.start()

        # Initial Setup
        send_command(ser, "STOP")
        time.sleep(0.5)
        send_command(ser, datetime.now().strftime("TIME %Y/%m/%d %H:%M:%S"))
        
        cmds = PARAMETERS * REPEAT_N if len(PARAMETERS) == 1 else PARAMETERS

        if len(PARAMETERS) == 1:
            send_command(ser, str(INIT - float(cmds[0])))

        send_command(ser, "GO")
        wait_for_state("available")

        for j, param in enumerate(cmds):
            print(f"\n--- CICLO {j+1}/{len(cmds)} (Param: {param}) ---")
            
            # 1. Cambia Parametro
            send_command(ser, param)
            if not wait_for_state("busy"): continue # Skip if failed
            if not wait_for_state("available"): continue

            # 2. Avvia Logger
            send_command(ser, "LOGGER_ON")
            # Arduino should go BUSY while opening file, then AVAILABLE
            wait_for_state("busy", timeout=2) 
            wait_for_state("available", timeout=2)

            print(f"Logging in corso per {LOG_DURATION}s...")
            for i in range(LOG_DURATION):
                time.sleep(1)
                print(f"Progress: {i+1}/{LOG_DURATION}s")

            # 3. Ferma Logger
            send_command(ser, "LOGGER_OFF")
            wait_for_state("busy", timeout=2)
            wait_for_state("available", timeout=2)

            print(f"Ciclo {param} completato.")

        print("\n🏁 Tutti i parametri completati.")
        send_command(ser, "STOP")
        time.sleep(1)
        stop_flag = True
        ser.close()

    except Exception as e:
        print(f"Errore: {e}")
        stop_flag = True

if __name__ == "__main__":
    main()