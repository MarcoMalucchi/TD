# -*- coding: utf-8 -*-
"""
[tdwf/device.py]
Controllo di base di Analog Discovery 2.
Include controllo del power supply.
"""

from sys import exit
import ctypes as ct              
from .config import dwf, constants  
import tdwf.tdwfconstants as tdwf

class AD2:
    
    def __init__(self, idevice: int = 1, iconfig: int = 1, autoconf: int = 3) -> None:   
        """
        **Connessione ad Analog Discovery 2**. Verifica se esiste e si connette con il 
        dispositivo richiesto. Una info chiave fornite da python, poi cruciale per 
        avviare oscilloscopio, genertore di funzioni, ecc è

                `.hdwf`

        che è un *handle* (di fatto, un indice numerico, accessibile come `hdwf.value`) 
        che individua il dispositivo con cui si è stabilita la connessione. 

        Parameters
        ----------
        idevice : int     
            indice del dispositivo 
            [default 1, scelta giusta se c'è un solo device]
        iconfig : int
            indice della configurazione
            [default 1, vedere manuale SDK o WaveForms]
        autoconf : int   
            tipo di autoconfigurazione
                * 0 = disabilitata
                * 1 = abilitata
                * 3 = dinamica [default]

        Returns
        -------
        None

        Notes
        -----

        Esistono tre possibili outcomes dell'inizializzaione di classe
        1. **L'inizializzazione va a buon fine**. Il codice notifica sulla console
            * Conferma connessione
            * Numero di Serie del dispositivo
            * Numero di handle del dispositivo (tipicamente, 1)
            * Numero di configurazione scelta
        2. **Il dispositivo richiesto non viene trovato**. Il codice genera
        una eccezione di tipo Runtime con valore "Dispositivo non trovato"
        3. **Il dispositivo c'è ma non è accessibile**. Questo succede tipicamente
        quando è già collegato ad altro, per esempio a WaveForms. In codice generare
        una eccezione di tipo Runtime con valore "Dispositivo non raggiungibile".     
        """
        ndevice = ct.c_int()
        dwf.FDwfEnum(constants.enumfilterAll, ct.byref(ndevice))
        if idevice>ndevice.value:
            print(f"Dispositivo #{idevice} non presente...")
            raise RuntimeError("Dispositivo non trovato")
            #exit()
        else:
            self.hdwf = ct.c_int()
            res = dwf.FDwfDeviceConfigOpen(ct.c_int(idevice-1), ct.c_int(iconfig-1), ct.byref(self.hdwf))      
            if res == 0:
                dwf.FDwfDeviceCloseAll()
                res = dwf.FDwfDeviceConfigOpen(ct.c_int(idevice-1), ct.c_int(0), ct.byref(self.hdwf))
            if res == 0:
                print(f"Dispositivo #{idevice} non raggiungibile...")
                raise RuntimeError("Dispositivo non raggiungibile")
                #exit()            
            else:
                serialnum = ct.create_string_buffer(32) 
                dwf.FDwfEnumSN(ct.c_int(idevice-1), serialnum)
                self.SN = serialnum.value.decode("ascii")
                print(f'Dispositivo #{idevice} [{self.SN}, hdwf={self.hdwf.value}] connesso!')
                print(f'Configurazione #{iconfig}')
        dwf.FDwfDeviceAutoConfigureSet(self.hdwf,ct.c_int(autoconf))
        self._getinfo()

    def _getinfo(self) -> None:
        temp1 = ct.c_int()
        dwf.FDwfAnalogInChannelCount(self.hdwf, ct.byref(temp1))
        self.ain_nchans = temp1.value
        dwf.FDwfAnalogInBufferSizeInfo(self.hdwf, 0, ct.byref(temp1))
        self.ain_nmax = temp1.value
        dwf.FDwfAnalogInBitsInfo(self.hdwf, ct.byref(temp1))
        self.ain_nbits = temp1.value
        # Range
        temp1 = ct.c_double()
        temp2 = ct.c_double()
        temp3 = ct.c_double()
        dwf.FDwfAnalogInChannelRangeInfo(self.hdwf, ct.byref(temp1), 
            ct.byref(temp2), ct.byref(temp3))
        self.ain_rngmin = temp1.value
        self.ain_rngmax = temp2.value
        self.ain_nrng = int(temp3.value)
        # Offset
        dwf.FDwfAnalogInChannelOffsetInfo(self.hdwf, ct.byref(temp1), 
            ct.byref(temp2), ct.byref(temp3))
        self.ain_offmin = temp1.value
        self.ain_offmax = temp2.value
        self.ain_noff = int(temp3.value)
        # Proprietà dei canali output  
        temp1 = ct.c_int()
        dwf.FDwfAnalogOutCount(self.hdwf, ct.byref(temp1))
        self.aout_nchans = temp1.value 
        dwf.FDwfAnalogOutNodeInfo(self.hdwf, ct.c_int(0), ct.byref(temp1))
        self.aout_nodes = temp1.value  
        # Digital
        temp1 = ct.c_double()
        dwf.FDwfDigitalOutInternalClockInfo(self.hdwf, ct.byref(temp1)) 
        self.dclk = temp1.value
        temp1 = ct.c_int()
        dwf.FDwfDigitalOutCount(self.hdwf, ct.byref(temp1))
        self.dout_nchans = temp1.value 
        dwf.FDwfDigitalInBufferSizeInfo(self.hdwf, ct.byref(temp1))
        self.din_nmax = temp1.value

    def close(self) -> None:    
        """
        Chiude la comunicazione con Analog Discovery 2. Necessario per poi
        usare il dispositivo con altri software, per esempio Waveforms.
        """
        dwf.FDwfDeviceClose(self.hdwf)    
        print("Dispositivo disconnesso.")    

    @property
    def error(self) -> tuple[int, str]:
        """
        Riporta l'ultimo errore registrato da Analog Discovery 2. In assenza di errori
        il codice è zero.

        Returns
        -------
        int
            stringa di errore
        str
            codice di errore

        Examples
        --------
        >>> mioad2 = tdwf.AD()
        >>> [...]
        >>> errcode,errstr = mioad2.error
        >>> if errcode==0:
        >>>     print("Nessun errore)
        >>> else:
        >>>     print(f"[Errore {errcode}]: {errstr}")
        """
        dfwerc = ct.c_int()
        dwf.FDwfGetLastError(ct.byref(dfwerc))
        errormsg = ct.create_string_buffer(512)  
        dwf.FDwfGetLastErrorMsg(errormsg)                  
        return dfwerc.value, errormsg.value.decode("ascii")

    @property
    def temp(self) -> float:
        """
        Misura e fornisce la temperatura di Analog Discovery 2.

        Returns
        -------
        float
            valore della temperatura in gradi

        Examples
        --------
        >>> mioad2 = tdwf.AD()
        >>> prinnt("Temperatura AD2 = {mioad2.temp}C")
        """

        dwf.FDwfAnalogIOStatus(self.hdwf) # => Effettua misura
        temp = ct.c_double()
        dwf.FDwfAnalogIOChannelNodeStatus(self.hdwf, tdwf.chan_usb, tdwf.usb_T, ct.byref(temp))
        return temp.value

    @property
    def vdd(self) -> float:
        return self._vdd # No check su dispositivo (!)
    @vdd.setter
    def vdd(self,value: float) -> float:
        dwf.FDwfAnalogIOChannelNodeSet(self.hdwf, tdwf.chan_vdd, tdwf.vdd_enab, ct.c_bool(True))
        dwf.FDwfAnalogIOChannelNodeSet(self.hdwf, tdwf.chan_vdd, tdwf.vdd_V, ct.c_double(value))
        self._vdd = value 
        return value

    @property
    def vss(self) -> float:
        return self._vss # No check su dispositivo (!)
    @vss.setter
    def vss(self,value: float) -> float:
        dwf.FDwfAnalogIOChannelNodeSet(self.hdwf, tdwf.chan_vss, tdwf.vss_enab, ct.c_bool(True))
        dwf.FDwfAnalogIOChannelNodeSet(self.hdwf, tdwf.chan_vss, tdwf.vss_V, ct.c_double(value))
        self._vss = value
        return value

    def power(self,enable: bool) -> None:
        """
        Accende/spegne il power supply duale in base alle impostazioni.

        Parameters
        ----------
        enable : bool 
            indica se accendere o spegnere il power supply

        Returns
        -------
        None

        Examples
        --------
        Esempio di utilizzo: imposta duale a +/-3V e la accende
        
        >>> mioad2 = tdwf.AD()
        >>> mioad2.vdd = +3
        >>> mioad2.vss = -3
        >>> mioad2.power(True)
        """
        dwf.FDwfAnalogIOEnableSet(self.hdwf, ct.c_bool(enable))  

    def reset(self) -> None:
        """
        Reset generale del dispositivo e mette configurazione di default
        """
        dwf.FDwfDeviceReset(self.hdwf)
        self._getinfo() # serve?                     
