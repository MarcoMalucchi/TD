import serial
import time
import threading
from datetime import datetime

# ==== CONFIGURAZIONE ====
PORT = '/dev/ttyACM0' #su Linux/Mac
BAUD = 9600
PARAMETERS = ["+0.25"]    # o ["1.000", "2.000", "3.000"]
INIT = 1.5                # frequenza iniziale usato solo se PARAMETERS ha un solo elemento
REPEAT_N = 2              # usato solo se PARAMETERS ha un solo elemento
LOG_DURATION = 5
# =========================

arduino_state = "unknown"
stop_flag = False
lock = threading.Lock()  # per proteggere accesso allo stato


def serial_listener(ser):
    """Thread che ascolta continuamente la seriale e stampa i messaggi in tempo reale."""
    global arduino_state, stop_flag
    while not stop_flag:
        if ser.in_waiting > 0:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                # stampa immediata di qualsiasi messaggio
                print(f"{line}")

                # aggiorna lo stato globale se è un messaggio noto
                low = line.lower()
                if low == "busy" or low == "available":
                    with lock:
                        arduino_state = low
        time.sleep(0.01)


def wait_for_state(state):
    """Attende finché Arduino non entra nello stato richiesto."""
    print(f"⏳ Attendo stato '{state}'...")
    while True:
        with lock:
            if arduino_state == state:
                print(f"Arduino è '{state}'.")
                return
        if stop_flag:
            return
        time.sleep(0.05)


def send_command(ser, cmd):
    """Invia un comando ad Arduino."""
    print(f"Invio comando: {cmd}")
    ser.write((cmd + "\n").encode())


def main():
    global stop_flag

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)
        print(f"Connesso su {PORT} a {BAUD} baud.")

        #Normalizzazione parametri
        num_par = len(PARAMETERS)
        if num_par == 1 and REPEAT_N > 1:
            print(f"Ripeto comando '{PARAMETERS[0]}' {REPEAT_N} volte.")
            cmds = PARAMETERS * REPEAT_N
        else:
            cmds = PARAMETERS

        #Avvia il listener in background
        listener = threading.Thread(target=serial_listener, args=(ser,), daemon=True)
        listener.start()

        send_command(ser, "STOP")
        send_command(ser, datetime.now().strftime("TIME %Y/%m/%d %H:%M:%S"))
        if num_par == 1:
            send_command(ser, str(INIT-float(cmds[0])))  #Frequenza iniziale
        num_par = len(cmds)
        send_command(ser, "GO")     #Avvio motore

        # Attende che Arduino sia pronto
        wait_for_state("available")

        j = 0
        for param in cmds:
            print(f"\nInizio ciclo per parametro: {param}")

            #invia comando di cambio parametro
            print (f"iterazione n. {j+1} di {num_par}")
            tf = round((num_par-j)*(LOG_DURATION+1)/60, 2)
            print(f"Tempo rimanente stimato {tf} min.")
            send_command(ser, param)
            wait_for_state("busy")
            wait_for_state("available")

            #avvia logging
            send_command(ser, "ON_LOGGER")
            #wait_for_state("busy")

            #lascia lavorare Arduino per LOG_DURATION secondi
            print(f"Logging in corso per {LOG_DURATION} s...")
            for i in range(LOG_DURATION):
                time.sleep(1)
                print(f"{i+1}/{LOG_DURATION}s")

            #ferma logging
            send_command(ser, "OFF_LOGGER")
            time.sleep(1)
            #wait_for_state("busy")
            #wait_for_state("available")

            print(f"Ciclo {param} completato.")
            j += 1

        print("\nTutti i parametri completati.")
        send_command(ser, "STOP")
        stop_flag = True
        ser.close()

    except serial.SerialException as e:
        print(f"Errore seriale: {e}")
    except KeyboardInterrupt:
        stop_flag = True
        print("\nInterrotto dall’utente.")


if __name__ == "__main__":
    main()
