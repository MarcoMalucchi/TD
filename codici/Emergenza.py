import tdwf
import time

ad2 = tdwf.AD2()
ad2.vdd = 5.0
ad2.power(True)
time.sleep(0.2)

i2c = tdwf.I2Cbus(ad2.hdwf)

# The two possible addresses for MPU-6050
possible_addresses = [0x68, 0x69]
who_am_i_reg = 0x75

print("Searching for MPU-6050...")

for addr in possible_addresses:
    try:
        # We write the register address (0x75) then read 1 byte back
        data = i2c.read(addr, who_am_i_reg, 1)
        if data and data[0] == 0x68:
            print(f"SUCCESS! Sensor found at 0x{addr:02x}")
            print(f"WHO_AM_I register returned: 0x{data[0]:02x}")
            break
    except:
        continue
else:
    print("Could not find sensor. Check if SDA/SCL are swapped (Pin 0 vs Pin 1).")

ad2.close()