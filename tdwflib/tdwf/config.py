# -*- coding: utf-8 -*-
"""
[tdwf/config.py]
Questo modulo serve ad accedere alla SDK in una maniera che è indipendente 
dal tipo di sistema operativo utilizzato (cross-platform). La SDK Digilent e
questo wrapper al momento supportano

    * windows
    * linux
    * mac os
"""

import ctypes as ct                   
from sys import platform, path, exit
from os import sep   

if platform.startswith("win"): # Win
    dwf = ct.cdll.dwf
    constants_path = "C:" + sep + "Program Files (x86)" + sep + "Digilent" \
        + sep + "WaveFormsSDK" + sep + "samples" + sep + "py"
elif platform.startswith("darwin"): # MacOS
    lib_path = sep + "Library" + sep + "Frameworks" + sep + "dwf.framework" \
        + sep + "dwf"
    dwf = ct.cdll.LoadLibrary(lib_path)
    constants_path = sep + "Applications" + sep + "WaveForms.app" + sep + \
        "Contents" + sep + "Resources" + sep + "SDK" + sep + "samples" + sep + "py"
else:
    dwf = ct.cdll.LoadLibrary("libdwf.so")
    constants_path = sep + "usr" + sep + "share" + sep + "digilent" + sep \
        + "waveforms" + sep + "samples" + sep + "py"

path.append(constants_path)
import dwfconstants as constants                
    
version = ct.create_string_buffer(32)  
dwf.FDwfGetVersion(version)                  
print("Digilent WaveForms SDK versione", version.value.decode("ascii"))
