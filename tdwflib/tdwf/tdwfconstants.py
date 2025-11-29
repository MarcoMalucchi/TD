# -*- coding: utf-8 -*-
"""
[tdwf/tdwfconstants.py]
Costanti utili per il controllo di Analog Discovery 2
"""

import ctypes as ct     
from .config import dwf, constants                

# -----------------------------------------------------------------------------
# Channel constants
# -----------------------------------------------------------------------------
chanCh1 = ct.c_int(0) 
chanCh2 = ct.c_int(1)
chanW1  = ct.c_int(0)
chanW2  = ct.c_int(1)
nodeCA  = ct.c_int(0)
nodeFM  = ct.c_int(1)
nodeAM  = ct.c_int(2)
D00 = 0x0001
D01 = 0x0002
D02 = 0x0004
D03 = 0x0008
D04 = 0x0010
D05 = 0x0020
D06 = 0x0040
D07 = 0x0080
D08 = 0x0100
D09 = 0x0200
D10 = 0x0400
D11 = 0x0800
D12 = 0x1000
D13 = 0x2000
D14 = 0x4000
D15 = 0x8000
# -----------------------------------------------------------------------------
# Nodi power supply e temperatura
# -----------------------------------------------------------------------------
chan_vdd = ct.c_int(0)
vdd_enab = ct.c_int(0)
vdd_V = ct.c_int(1)
vdd_I = ct.c_int(2)
# -----------------------------------------------------------------------------
chan_vss = ct.c_int(1)
vss_enab = ct.c_int(0)
vss_V = ct.c_int(1)
vss_I = ct.c_int(2)
# -----------------------------------------------------------------------------
chan_usb = ct.c_int(2)
usb_V = ct.c_int(0)
usb_I = ct.c_int(1)
usb_T = ct.c_int(2)
# -----------------------------------------------------------------------------
# funzioni
# -----------------------------------------------------------------------------
funcCustom = constants.funcCustom
funcSine = constants.funcSine
funcSquare = constants.funcSquare
funcTriangle = constants.funcTriangle
funcNoise = constants.funcNoise
funcDC = constants.funcDC
funcPulse = constants.funcPulse
funcTrapezium = constants.funcTrapezium
funcSinePower = constants.funcSinePower
funcRampUp = constants.funcRampUp
funcRampDown = constants.funcRampDown
# -----------------------------------------------------------------------------
# trigger
# -----------------------------------------------------------------------------
trigslopeRise = constants.DwfTriggerSlopeRise
trigslopeFall = constants.DwfTriggerSlopeFall
trigslopeEither = constants.DwfTriggerSlopeEither
trigtypeEdge = constants.trigtypeEdge
trigtypePulse = constants.trigtypePulse
trigtypeTransition = constants.trigtypeTransition
trigtypeWindow = constants.trigtypeWindow
trigsrcNone = constants.trigsrcNone
trigsrcAIn = constants.trigsrcDetectorAnalogIn
trigsrcCh1 = ct.c_ubyte(101) # Custom constant
trigsrcCh2 = ct.c_ubyte(102) # Custom constant
trigsrcW1 = constants.trigsrcAnalogOut1
trigsrcW2 = constants.trigsrcAnalogOut2
trigsrcT1 = constants.trigsrcExternal1
trigsrcT2 = constants.trigsrcExternal2
# -----------------------------------------------------------------------------
# state
# -----------------------------------------------------------------------------
stateReady = constants.DwfStateReady
stateConfig = constants.DwfStateConfig
statePrefill = constants.DwfStatePrefill
stateArmed = constants.DwfStateArmed
stateWait = constants.DwfStateWait
stateTriggered = constants.DwfStateTriggered
stateRunning = constants.DwfStateRunning
stateNotDone = constants.DwfStateNotDone
stateDone = constants.DwfStateDone
# -----------------------------------------------------------------------------
