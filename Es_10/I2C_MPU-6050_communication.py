import tdwf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import time

#=======================================================
# ELENCO REGISTRI MPU6050 UTILIZZATI CON LORO INDIRIZZI.
#=======================================================

PWR_MGMT1 = 0x6B
PWR_MNGMT2 = 0x6C
WHO_AM_I = 0x75
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
TEMP_OUT_H = [0x41]
TEMP_OUT_L = [0x42]
ACCEL_XOUT_H = [0x3B]
ACCEL_XOUT_L = [0x3C]
ACCEL_YOUT_H = [0x3D]
ACCEL_YOUT_L = [0x3E]
ACCEL_ZOUT_H = [0x3F]
ACCEL_ZOUT_L = [0x40]
GYRO_XOUT_H = [0x43]
GYRO_XOUT_L = [0x44]
GYRO_YOUT_H = [0x45]
GYRO_YOUT_L = [0x46]
GYRO_ZOUT_H = [0x47]
GYRO_ZOUT_L = [0x48]
SAD = 0x68

#####################################

#====================
# CONFIGURAZIONE AD2.
#====================

# Inizializzazione
ad2 = tdwf.AD2()
# Alimentazione
ad2.vdd = 3.3
ad2.power(True)
# Inizializzazione bus I2C
i2c = tdwf.I2Cbus(ad2.hdwf)  # default 100kHz, SCL = D00, SDA = D01
# Ricerca dispositivi connessi al bus I2C
devs = i2c.scan()  # verifica cosa è connesso...
for dev in devs:
    print(f"Device: 0x{dev:02x}")
# Apertura comunicazione I2C AD2-MPU-6050 (con AD2 come master)
sht = tdwf.I2Cdevice(ad2.hdwf,SAD)

#####################

#=========================
# MPU-6050 CONFIGURATION.
#=========================

sht.write([PWR_MGMT1, 0x80])  # reset del sensore, faccio passare un po' di tempo perchè si riavvii nel modo corretto
time.sleep(0.7)
sht.write([PWR_MGMT1, 0x01])  # sveglia il sensore e seleziona clock, abilita sesore di temperatura
print("Sensor restarted")
time.sleep(0.5)
sht.write([PWR_MNGMT2, 0x00])   # abilita accelerometro e giroscopio
time.sleep(0.1)
sht.write([CONFIG, 0x03])       # filtro digitale
time.sleep(0.1)
sht.write([SMPLRT_DIV, 0x04])   # sample rate = Gyro output rate / (1 + SMPLRT_DIV)
time.sleep(0.1)
sht.write([GYRO_CONFIG, 0x08])  # full scale = +/- 500 deg/s   SENTIVITY SCALE FACTIO = 65.5 LSB/deg/s
time.sleep(0.1)
sht.write([ACCEL_CONFIG, 0x00]) # full scale = +/- 2g   SENTIVITY SCALE FACTIO = 16384 LSB/g
time.sleep(0.1)

# def raw_dump():
#     print("\n--- MPU-6050 FULL REGISTER DUMP ---")
#     print("    0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F")
#     for row in range(0, 0x76, 16):
#         line_vals = []
#         for col in range(16):
#             reg = row + col
#             if reg > 0x75: break
#             sht.writeread([reg], 1)
#             line_vals.append(f"{sht.vals[0]:02X}")
#         print(f"{row:02X}: {' '.join(line_vals)}")
#     print("------------------------------------\n")

# # Call this right after "Sensor restarted"
# raw_dump()

##########################

#=======================
# MEASUREMENT FUNCTIONS.
#=======================

# --- TEMPERATURE MEASURE ---
def read_temp():
    global start_time
    sht.writeread(TEMP_OUT_H, 2)
    raw_data = list(sht.vals)
    current_time = time.time() - start_time
    T = (int(raw_data[0]) << 8) | int(raw_data[1])
    if T > 32767: # Complemento a due
        T -= 65536
    T = T/340 + 36.53 # Conversione temperatura
    return T, current_time # Returns temperature in Celsius

# --- ACCELERATION MEASURE ---
def read_accel():
    global start_time
    sht.writeread(ACCEL_XOUT_H,6)
    #print(f"DEBUG: Rax vals {sht.vals}")
    raw_data = list(sht.vals)
    current_time = time.time() - start_time
    ACCEL = []
    # Check the Z-axis bytes specifically (index 4 and 5)
    # z_high = raw_data[4]
    # z_low = raw_data[5]
    # z_combined = (z_high << 8) | z_low
    # print(f"DEBUG Z-AXIS: High={hex(z_high)}, Low={hex(z_low)}, Combined={z_combined}")

    for i in range(0, 6, 2):    #l'indice itera di 2 in 2, quindi salta i numeri dispari
        value = (int(raw_data[i]) << 8) | int(raw_data[i+1])
        if value > 32767: # Complemento a due
            value -= 65536
        ACCEL.append(value / 16384.0)  # Scale factor for +/- 2g (to change it go to ACCEL_CONFIG register)

    #print(f"DEBUG: Cleaned value {ACCEL}")
    
    return ACCEL, current_time  # Returns [ACCEL_X, ACCEL_Y, ACCEL_Z] acceleration vector in g and current_time

# --- GYROSCOPE MEASURE ---
def read_gyro():
    global start_time
    sht.writeread(GYRO_XOUT_H,6)
    raw_data = list(sht.vals)
    current_time = time.time()-start_time
    GYRO = []

    for i in range(0, 6, 2):    #l'indice itera di 2 in 2, quindi salta i numeri dispari
        value = (int(raw_data[i]) << 8) | int(raw_data[i+1])
        if value > 32767: # Complemento a due
            value -= 65536
        GYRO.append(value / 65.5)  # Scale factor for +/- 2000 deg/s (to change it go to GYRO_CONFIG register)

    return GYRO, current_time  # Returns [GYRO_X, GYRO_Y, GYRO_Z] angular velocity vector in deg/s and current time

###########################

#=======================
# FUNZIONI DI LIVE PLOT.
#=======================

# --- LIVE PLOT TEMPERATURA ---

def setup_live_plot_T():
    """Initializes the figure and returns the objects needed for updates."""
    fig, ax = plt.subplots(figsize=(12, 6))
    # Create an empty line object
    line, = ax.plot([], [], ".-", color="tab:orange")
    plt.ylabel("Temperature [C]")
    plt.xlabel("Time [s]")
    plt.title("Press ESC to exit")
    plt.grid(True)
    
    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show(block=False)
    
    return fig, ax, line

def update_live_plot_T(fig, ax, line, x_data, y_data):
    """Updates the data in the plot and refreshes the canvas."""
    line.set_data(x_data, y_data)
    ax.relim()           
    ax.autoscale_view() 
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# --- LIVE PLOT ACCELERAZIONE ---

def setup_live_plot_A():
    """Initializes the figure and returns the objects needed for updates."""
    fig, ax = plt.subplots(3,1, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    # Create an empty line object
    line1, = ax[0].plot([], [], ".-", color="tab:blue", label="X axis")
    line2, = ax[1].plot([], [], ".-", color="tab:green", label="Y axis")
    line3, = ax[2].plot([],[], ".-", color="tab:red", label="Z axis")
    for a, label in zip([ax[0], ax[1], ax[2]], ["X acceleration [g]", "Y acceleration [g]", "Z acceleration [g]"]):
        a.set_ylabel(label)
        a.grid(True)
        a.set_ylim(-2, 2) # Fixed range prevents constant jumping
        a.legend(loc="upper right")
    plt.xlabel("Time [s]")
    plt.title("Press ESC to exit")
    plt.grid(True)
    
    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show(block=False)
    
    return fig, ax, line1, line2, line3

def update_live_plot_A(fig, ax, line1, line2, line3, x_data, y_Xdata, y_Ydata, y_Zdata):
    """Updates the data in the plot and refreshes the canvas."""
    line1.set_data(x_data, y_Xdata)
    line2.set_data(x_data, y_Ydata)
    line3.set_data(x_data, y_Zdata)
    for a in ax:
        a.relim()
        a.autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# --- LIVE PLOT GIROSCOPIO ---

def setup_live_plot_G():
    """Initializes the figure and returns the objects needed for updates."""
    fig, ax = plt.subplots(3,1, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    # Create an empty line object
    line1, = ax[0].plot([], [], ".-", color="tab:blue", label="X axis")
    line2, = ax[1].plot([], [], ".-", color="tab:green", label="Y axis")
    line3, = ax[2].plot([],[], ".-", color="tab:red", label="Z axis")
    for a, label in zip([ax[0], ax[1], ax[2]], ["X Gyro [deg/s]", "Y Gyro [deg/s]", "Z Gyro [deg/s]"]):
        a.set_ylabel(label)
        a.grid(True)
        a.set_ylim(-500, 500) # Fixed range prevents constant jumping
        a.legend(loc="upper right")
    plt.xlabel("Time [s]")
    plt.title("Press ESC to exit")
    plt.grid(True)
    
    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show(block=False)
    
    return fig, ax, line1, line2, line3

def update_live_plot_G(fig, ax, line1, line2, line3, x_data, y_Xdata, y_Ydata, y_Zdata):
    """Updates the data in the plot and refreshes the canvas."""
    line1.set_data(x_data, y_Xdata)
    line2.set_data(x_data, y_Ydata)
    line3.set_data(x_data, y_Zdata)
    for a in ax:
        a.relim()
        a.autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

########################

#=============================
# FUNZIONI DI GESTIONE EVENTI.
#=============================

def on_close(event):
    global flag_run
    flag_run = False

def on_key(event):
    global flag_run
    if event.key == 'escape':  # termina programma
        flag_run = False

###############################

#=================
# CICLO DI MISURA.
#=================

start_time = time.time()

try:
    while True:

        mytime = []
        TTv = []
        ACCEL = []
        GYRO = []

        flag_run = True
        flag_first = True

        choice = input("Select measurement type - Temperature (T), Acceleration (A), Gyroscope (G), or Quit (Q): ").strip().upper()

        if choice == 'T':
            fig, ax, hp1 = setup_live_plot_T()
            while flag_run:
                time.sleep(0.05)
                new_T, new_t = read_temp()
                mytime.append(new_t)
                TTv.append(new_T)
                update_live_plot_T(fig, ax, hp1, mytime[-100:], TTv[-100:])
            plt.close(fig)
        elif choice == 'A':
            fig, ax, hp1, hp2, hp3 = setup_live_plot_A()
            while flag_run:
                time.sleep(0.05)
                new_ACCEL, new_t = read_accel()
                mytime.append(new_t)
                ACCEL.append(new_ACCEL)
                x_vals = [row[0] for row in ACCEL[-100:]]
                y_vals = [row[1] for row in ACCEL[-100:]]
                z_vals = [row[2] for row in ACCEL[-100:]]
                update_live_plot_A(fig, ax, hp1, hp2, hp3, mytime[-100:], x_vals, y_vals, z_vals)  # Aggiorna il grafico con i dati di accelerazione
            plt.close(fig)
        elif choice == 'G':
            fig, ax, hp1, hp2, hp3 = setup_live_plot_G()
            while flag_run:
                time.sleep(0.05)
                new_GYRO, new_t = read_gyro()
                mytime.append(new_t)
                GYRO.append(new_GYRO)
                x_vals = [row[0] for row in GYRO[-100:]]
                y_vals = [row[1] for row in GYRO[-100:]]
                z_vals = [row[2] for row in GYRO[-100:]]
                update_live_plot_G(fig, ax, hp1, hp2, hp3, mytime[-100:], x_vals, y_vals, z_vals)
            plt.close(fig)
        elif choice == 'Q':
            break
        else:
            print("Invalid choice. Please enter T, A, G, or Q.")
finally:
    # Cleanup
    print("\nCleaning up hardware...")
    plt.close('all')
    ad2.power(False)
    ad2.close()
    print("AD2 Power Off. Program terminated safely.")
    
