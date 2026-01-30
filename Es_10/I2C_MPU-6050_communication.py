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
# CONFIGURAZIONE MPU-6050.
#=========================

sht.write([PWR_MGMT1, 0x80])  # reset del sensore, faccio passare un po' di tempo perchè si riavvii nel modo corretto
time.sleep(1)
sht.write([PWR_MGMT1, 0x00])  # sveglia il sensore e seleziona clock, abilita sesore di temperatura
time.sleep(0.5)
sht.write([PWR_MNGMT2, 0x00])   # abilita accelerometro e giroscopio
time.sleep(0.1)
sht.write([CONFIG, 0x03])       # filtro digitale
time.sleep(0.1)
sht.write([SMPLRT_DIV, 0x04])   # sample rate = Gyro output rate / (1 + SMPLRT_DIV)
time.sleep(0.1)
sht.write([GYRO_CONFIG, 0x08])  # full scale = +/- 2000 deg/s   SENTIVITY SCALE FACTIO = 65.5 LSB/deg/s
time.sleep(0.1)
sht.write([ACCEL_CONFIG, 0x00]) # full scale = +/- 2g   SENTIVITY SCALE FACTIO = 16384 LSB/g
time.sleep(0.1)

##########################

###########################
# FUNZIONI DI LETTURA DATI.
###########################

# --- LETTURA TEMPPERATURA ---
def read_temp():
    sht.writeread(TEMP_OUT_H,2)
    T = (sht.vals[0] << 8) | sht.vals[1]
    if T>=32767:    # Complemento a due
        T-=65536
    T = T/340 + 36.53 # Conversione temperatura
    return T # Returns temperature in Celsius

# --- LETTURA ACCELERAZIONE ---
def read_accel():
    sht.writeread(ACCEL_XOUT_H,6)
    ACCEL = []

    for i in range(0, 6, 2):    #l'indice itera di 2 in 2, quindi salta i numeri dispari
        value = (sht.vals[i] << 8) | sht.vals[i+1]
        if value >= 0x8000:  # Complemento a due
            value -= 0x10000
        ACCEL.append(value / 16384.0)  # Scale factor for +/- 2g (to change it go to ACCEL_CONFIG register)
    
    return ACCEL  # Returns [ACCEL_X, ACCEL_Y, ACCEL_Z] acceleration vector in g

# --- LETTURA GIROSCOPIO ---
def read_gyro():
    sht.writeread(GYRO_XOUT_H,6)
    GYRO = []

    for i in range(0, 6, 2):    #l'indice itera di 2 in 2, quindi salta i numeri dispari
        value = (sht.vals[i] << 8) | sht.vals[i+1]
        if value >= 0x8000:  # Complemento a due
            value -= 0x10000
        GYRO.append(value / 65.5)  # Scale factor for +/- 2000 deg/s (to change it go to GYRO_CONFIG register)

    return GYRO  # Returns [GYRO_X, GYRO_Y, GYRO_Z] angular velocity vector in deg/s

###########################

#=======================
# FUNZIONI DI LIVE PLOT.
#=======================

# --- LIVE PLOT TEMPERATURA ---



# --- LIVE PLOT ACCELERAZIONE ---



# --- LIVE PLOT GIROSCOPIO ---

########################

    # -[Funzioni di gestione eventi]-----------------------------------------------

    def on_close(event):
        global flag_run
        flag_run = False

    def on_key(event):
        global flag_run
        if event.key == 'escape':  # termina programma
            flag_run = False


import tdwf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import time

# --- PLOT MANAGEMENT FUNCTIONS ---

def setup_live_plot():
    """Initializes the figure and returns the objects needed for updates."""
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.grid(True)
    plt.ylabel("Temperature [C]")
    plt.xlabel("Time [s]")
    plt.title("Press ESC to exit")
    
    # Create an empty line object
    line, = ax.plot([], [], "o-", color="tab:orange") 
    
    plt.tight_layout()
    plt.show(block=False)
    
    return fig, ax, line

def update_live_plot(fig, ax, line, x_data, y_data):
    """Updates the data in the plot and refreshes the canvas."""
    line.set_data(x_data, y_data)
    ax.relim()           
    ax.autoscale_view() 
    fig.canvas.draw()
    fig.canvas.flush_events()

# --- HARDWARE SETUP ---
# (Keeping your original MPU6050 and AD2 configuration)
PWR_MGMT1 = 0x6B
TEMP_OUT_H = [0x41]
TEMP_OUT_L = [0x42]
SAD = 0x68

ad2 = tdwf.AD2()
ad2.vdd = 3.3
ad2.power(True)
i2c = tdwf.I2Cbus(ad2.hdwf)
sht = tdwf.I2Cdevice(ad2.hdwf, SAD)
sht.write([PWR_MGMT1, 0x00])  

# --- EVENT HANDLERS ---
flag_run = True

def on_close(event):
    global flag_run
    flag_run = False

def on_key(event):
    global flag_run
    if event.key == 'escape':
        flag_run = False

# --- MEASUREMENT CYCLE ---

# 1. Initialize Plot using our function
fig, ax, hp1 = setup_live_plot()
fig.canvas.mpl_connect("close_event", on_close)
fig.canvas.mpl_connect("key_press_event", on_key)

mytime = []
TTv = []
start_time = time.time()

try:
    while flag_run:
        time.sleep(0.1)
        
        # Data Acquisition
        sht.writeread(TEMP_OUT_H, 1)
        TH = sht.vals[0]
        sht.writeread(TEMP_OUT_L, 1)
        TL = sht.vals[0]

        # Temperature Conversion
        TT = (TH << 8) + TL
        if TT > 0x8000:
            TT -= 0x10000
        TT = TT/340 + 36.53
        
        # Update Data Lists
        TTv.append(TT)
        mytime.append(time.time() - start_time)
        print(f"T = {TT:.2f}C")

        # 2. Call the Update function
        update_live_plot(fig, ax, hp1, mytime, TTv)

finally:
    # Cleanup
    plt.close(fig)
    ad2.close()
    print("Program terminated safely.")

    # -[Ciclo di misura]-----------------------------------------------------------
    #   1. Creazione figura e link agli eventi
    # fig, ax = plt.subplots(figsize=(12,6))
    # fig.canvas.mpl_connect("close_event", on_close)
    # fig.canvas.mpl_connect("key_press_event", on_key)

    #   2. Inizializzazione variabili - MISURA
    # mytime = []
    # TTv = []
    # flag_first = True
    # flag_run = True
    # ACCEL_X = []
    # ACCEL_Y = []
    # ACCEL_Z = []

    # start_time = time.time()

    #   3. Ciclo di misura
    # while flag_run:
    #     time.sleep(0.1)

    #     sht.writeread([WHO_AM_I], 1)
    #     if sht.vals[0] != 0x68:
    #         print(sht.vals[0])
    #         print("ALARM: Communication lost! Check wires.")
    #         continue # Skip this loop if sensor isn't responding

    #   Misura accelerazione.

        # sht.writeread(ACCEL_XOUT_H,6)
        # ACCEL_X.append((sht.vals[0] << 8 | sht.vals[1])/16384)
        # ACCEL_Y.append((sht.vals[2] << 8 | sht.vals[3])/16384)
        # ACCEL_Z.append((sht.vals[4] << 8 | sht.vals[5])/16384)



    #     sht.writeread(TEMP_OUT_H,2)
    #     TH = sht.vals[0]
    #     #sht.writeread(TEMP_OUT_L,1)
    #     TL = sht.vals[1]



    #     mytime.append(time.time()-start_time)
    #     if flag_first:
    #         flag_first = False
    #         hp1, = plt.plot(mytime,TTv, "o", color="tab:orange")
    #         plt.grid(True)
    #         plt.ylabel("Temperature [C]")
    #         plt.xlabel("Time [s]")
    #         plt.title("Press ESC to exit")
    #         plt.show(block=False)
    #         plt.tight_layout()
    #     else:
    #         hp1.set_ydata(TTv)
    #         hp1.set_xdata(mytime)
    #         ax.relim()           
    #         ax.autoscale_view() 
    #         fig.canvas.draw()
    #         fig.canvas.flush_events()
    # # ---------------------------------------
    # plt.close(fig)

    # -[Plot Setup]----------------------------------------------
    # Create a figure with 3 vertical subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    def on_key(event):
        global flag_run
        if event.key == 'escape':
            flag_run = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Data containers
    mytime, ACCEL_X, ACCEL_Y, ACCEL_Z = [], [], [], []
    flag_run = True
    start_time = time.time()

    # Initial plot lines (placeholders)
    lineX, = ax1.plot([], [], marker='o', linestyle='', color='tab:red', label='X-axis')
    lineY, = ax2.plot([], [], marker='o', linestyle='', color='tab:green', label='Y-axis')
    lineZ, = ax3.plot([], [], marker='o', linestyle='', color='tab:blue', label='Z-axis')

    for ax, label in zip([ax1, ax2, ax3], ["X [g]", "Y [g]", "Z [g]"]):
        ax.set_ylabel(label)
        ax.grid(True)
        ax.set_ylim(-2, 2) # Fixed range prevents constant jumping
        ax.legend(loc="upper right")

    ax3.set_xlabel("Time [s]")
    ax1.set_title("Live Acceleration Data (Press ESC to stop) [g]")

    plt.show(block=False)

    # -[Ciclo di misura]------------------------------------------
    while flag_run:
        time.sleep(0.05) # Slightly faster sampling for acceleration
        
        # Read 6 bytes starting from ACCEL_XOUT_H (X_H, X_L, Y_H, Y_L, Z_H, Z_L)
        sht.writeread(ACCEL_XOUT_H, 1)
        ACCEL_X_H = sht.vals[0]
        sht.writeread(ACCEL_XOUT_L, 1)
        ACCEL_X_L = sht.vals[0]
        sht.writeread(ACCEL_YOUT_H, 1)
        ACCEL_Y_H = sht.vals[0]
        sht.writeread(ACCEL_YOUT_L, 1)
        ACCEL_Y_L = sht.vals[0]
        sht.writeread(ACCEL_ZOUT_H, 1)
        ACCEL_Z_H = sht.vals[0]
        sht.writeread(ACCEL_ZOUT_L, 1)
        ACCEL_Z_L = sht.vals[0]
        
        # Helper function for Two's Complement and conversion
        def convert_accel(h, l):
            val = (h << 8) | l
            if val >= 0x8000:
                val -= 0x10000
            return val / 16384.0 # Scale factor for +/- 2g

        current_t = time.time() - start_time
        mytime.append(current_t)
        ACCEL_X.append(convert_accel(ACCEL_X_H, ACCEL_X_L))
        ACCEL_Y.append(convert_accel(ACCEL_Y_H, ACCEL_Y_L))
        ACCEL_Z.append(convert_accel(ACCEL_Z_H, ACCEL_Z_L))

        # Update plots
        lineX.set_data(mytime, ACCEL_X)
        lineY.set_data(mytime, ACCEL_Y)
        lineZ.set_data(mytime, ACCEL_Z)

        # Auto-scaling logic
        for ax in [ax1, ax2, ax3]:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()



    # This block executes even if you stop the kernel or an error occurs
    print("\nCleaning up hardware...")
    plt.close('all')
    ad2.power(False)
    ad2.close()
    print("AD2 Power Off. Process safely terminated.")