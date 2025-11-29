# -*- coding: utf-8 -*-
"""
[tdwf/protocols.py]
Gestione dei protocolli di comunicazione digitali. Per 
ora è completo solo I2C.
"""

import ctypes as ct              
import numpy as np
import time
from sys import exit
from .config import dwf, constants  
import tdwf.tdwfconstants as tdwf

class I2Cdevice:
    
    def __init__(self, hdwf: ct.c_int, sad: int) -> None:
        """
        Inizializza un dispositivo I2C.

        Parameters
        ----------
        hdwf : ct.c_int
            handle di Analog Discovery 2 connesso
        sad : int
            Slave ADdress del dispositivo

        Returns
        -------
        None
        """
        self.hdwf = hdwf
        self.sad8 = ct.c_int(sad << 1)
    
    def writeread(self, values: list[int], nbytes: int) -> int:
        """
        Scrive bytes sul dispositivo e legge delle risposte. 

        Parameters
        ----------
        values : List[int]
            Lista dei bytes da scrivere
        nbytes : int
            Numero di bytes da leggere come risposta

        Returns
        -------
        int
            valore Nak

        Notes
        -----

        Le risposte del dispositivo I2C non sono nel `return` ma sono bensì disponibili 
        nella proprietà `vals`. Esempio specifico in cui vengono scritti due bytes e 
        letti tre

        >>> iNak = dispI2C.writeread([0x10, 0x13],3)
        >>> if iNak==0:
        >>>     byte1 = dispI2C.vals[0]
        >>>     byte2 = dispI2C.vals[1]
        >>>     byte3 = dispI2C.vals[2]
        >>> else:
        >>>     print("Comunicazione fallita")
        """
        iNak = ct.c_int()
        nw = len(values)
        self._bufw = (ct.c_ubyte * nw)()
        self._bufr = (ct.c_ubyte * nbytes)()
        self.vals = np.frombuffer(self._bufr, dtype = np.uint8)
        for ii, dd in enumerate(values):
            self._bufw[ii] = dd    
        dwf.FDwfDigitalI2cWriteRead(self.hdwf, self.sad8, self._bufw, ct.c_int(nw), 
                                    self._bufr, ct.c_int(nbytes), ct.byref(iNak))
        return (iNak.value == 1)
        
    def read(self, nbytes: int) -> int: 
        """
        Legge `nbytes` bytes dal dispositivo.

        Parameters
        ----------
        nbytes : int
            Numero di bytes da leggere come risposta

        Returns
        -------
        int
            valore Nak

        Notes
        -----

        Le risposte del dispositivo I2C non sono nel `return` ma sono bensì disponibili 
        nella proprietà `vals`. Esempio specifico in cui letti due bytes

        >>> iNak = dispI2C.read(2)
        >>> if iNak==0:
        >>>     byte1 = dispI2C.vals[0]
        >>>     byte2 = dispI2C.vals[1]
        >>> else:
        >>>     print("Comunicazione fallita")
        """
        iNak = ct.c_int()
        self._bufr = (ct.c_ubyte * nbytes)()
        self.vals = np.frombuffer(self._bufr, dtype = np.uint8)
        dwf.FDwfDigitalI2cRead(self.hdwf, self.sad8, self._bufr, ct.c_int(nbytes), ct.byref(iNak))
        return (iNak.value == 1)
        
    def write(self, values: int | list[int]) -> int:
        """
        Scrive bytes sul dispositivo.

        Parameters
        ----------
        values : int oppure List[int]
            Singolo byte o lista dei bytes da scrivere

        Returns
        -------
        int
            valore Nak

        Examples
        --------

        Scrittura singolo byte

        >>> dispI2C.write(0x10)

        Scrittura di 3 bytes: 0x10, ossia 16 in decimale, 0x11 e 114.

        >>> dispI2C.write([0x10, 0x11, 114])
        """
        iNak = ct.c_int()
        if isinstance(values, int):
            dwf.FDwfDigitalI2cWriteOne(self.hdwf, self.sad8, ct.c_ubyte(values), ct.byref(iNak))
        else:
            nw = len(values)
            self._bufw = (ct.c_ubyte * nw)()
            for ii, dd in enumerate(values):
                self._bufw[ii] = dd    
            dwf.FDwfDigitalI2cWrite(self.hdwf, self.sad8, self._bufw, ct.c_int(nw), ct.byref(iNak))
        return (iNak.value == 1)
    
class I2Cbus:

    def __init__(self, hdwf: ct.c_int, scl: int = 0, sda: int = 1, 
                 rate: float = 100e3, stre: bool = True) -> None:   
        """
        Inizializza il bus I2C.
        
        Parameters
        ----------
        hdwf : ct.c_int
            Handle al dispositivo Analog Discovery 2 connesso
        scl : int
            Numero linea digitale di Serial CLock
            [default 0]
        sda : int
            Numero linea digitale di Serial DAta
            [default 1]
        rate : float
            Frequenza di clock pe ril bus I2C
            [default 100kHz]
        stre : bool
            Abilitazione clock stretching
            [default `True`]

        Returns
        -------
        None
        """
        self.hdwf = hdwf
        self.reset()
        self.rate = rate
        self.scl = scl
        self.sda = sda
        self.stre = stre
        self.clear()
    
    @property
    def rate(self) -> float:
        return self._rate
    @rate.setter
    def rate(self, value: float) -> float:
        dwf.FDwfDigitalI2cRateSet(self.hdwf, ct.c_double(value))
        self._rate = value
        return value
    
    @property
    def scl(self) -> int:
        return self._scl
    @scl.setter
    def scl(self, value: int) -> int:
        dwf.FDwfDigitalI2cSclSet(self.hdwf, ct.c_int(value))
        self._scl = value
        return value
    
    @property
    def sda(self) -> int:
        return self._sda
    @sda.setter
    def sda(self, value: int) -> int:
        dwf.FDwfDigitalI2cSdaSet(self.hdwf, ct.c_int(value))
        self._sda = value
        return value
    
    @property
    def stre(self) -> bool:
        return self._stre
    @stre.setter
    def stre(self, value: bool) -> bool:
        if value:
            dwf.FDwfDigitalI2cStretchSet(self.hdwf, ct.c_int(1))
        else:
            dwf.FDwfDigitalI2cStretchSet(self.hdwf, ct.c_int(0))
        self._stre = value
        return value

    def clear(self) -> None:
        """
        Avvio/verifica del bus I2C. Se il bus non risponde bene (i pull-up sono ok?
        l'alimentazione del dispositivo è accesa?), viene generata una eccezione 
        di tipo Runtime con valore "Avvio I2C fallito".
        """
        iNak = ct.c_int(0)
        count = 0
        while iNak.value == 0:
            dwf.FDwfDigitalI2cClear(self.hdwf, ct.byref(iNak))
            time.sleep(0.01)
            count += 1
            if count > 200:
                print("Avvio I2C fallito...")            
                raise RuntimeError("Avvio I2C fallito")
                #exit()
        print("Bus I2C pronto...")
            
    def reset(self) -> None:
        """
        Reset del bus I2C
        """
        dwf.FDwfDigitalI2cReset(self.hdwf)
        
    def scan(self) -> list[int]:
        """
        Fa una scansione del bus I2C per trovare i dispositivi connessi

        Returns
        -------
        List[int]        
            Lista di indirizzi `SAD` che hanno dato segni di vita
        """
        iNak = ct.c_int()
        devs = []
        for sad in range(1,127):
            dwf.FDwfDigitalI2cWrite(self.hdwf, ct.c_int(sad<<1), 0, ct.c_int(0), ct.byref(iNak))
            if iNak.value == 0:
                devs.append(sad)
        return devs        
        
# Non implementati...
class SPIdevice:
    
    def __init__(self):
        pass

class SPIbus:
    
    def __init__(self):
        pass