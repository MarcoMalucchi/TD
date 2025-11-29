# -*- coding: utf-8 -*-
"""
[tdwf/analogin.py]
Gestione dell'oscilloscopio Analogico
"""

import ctypes as ct        
import time   
import datetime
import numpy as np  
from .config import dwf, constants    
import tdwf.tdwfconstants as tdwf  
    
class ChanIn:
    
    def __init__(self, hdwf: ct.c_int, _chan, rng: float = 5.0, offs: float = 0.0, 
                 enab: bool = True, avg: bool = False) -> None:
        """
        Classe riservata... 
        """
        self._chan = _chan
        self.hdwf = hdwf
        self.rng = rng
        self.offs = offs
        self.enab = enab
        self.avg = avg
        
    """
    Notare: 
        AD2 accetta solo due range, 5 e 50V. Tuttavia, se il parametro viene
        letto di nuovo da AD2 si ottiene in valore di calibrazione, che sarà
        vicino a 5 o 50, ma non esattamente coincidente con questi valori.
    """
    @property 
    def rng(self) -> float:
        return self._rng
    @rng.setter
    def rng(self, value: float) -> float:
        tmp = ct.c_double()
        dwf.FDwfAnalogInChannelRangeSet(self.hdwf, self._chan, ct.c_double(value))
        dwf.FDwfAnalogInChannelRangeGet(self.hdwf, self._chan, ct.byref(tmp))
        self._rng = tmp.value
        return value
    
    """
    Funzionamento analogo a quello del range
    """
    @property 
    def offs(self) -> float:
        return self._offs
    @offs.setter
    def offs(self, value: float) -> float:
        tmp = ct.c_double()
        dwf.FDwfAnalogInChannelOffsetSet(self.hdwf, self._chan, ct.c_double(value))
        dwf.FDwfAnalogInChannelOffsetGet(self.hdwf, self._chan, ct.byref(tmp))
        self._offs = tmp.value
        return value     
    
    @property 
    def npt(self) -> int:
        return self._npt
    @npt.setter
    def npt(self, value: int) -> int:
        # Crea buffer binari e li linka ad array risultati
        self._buffer = (ct.c_double * value)()
        self.vals = np.frombuffer(self._buffer, dtype = np.float64)
        self._buffer16 = (ct.c_int16 * value)()
        self.vals16 = np.frombuffer(self._buffer16, dtype = np.int16)
        self._bufmin = (ct.c_double * (value >> 3))()
        self.min = np.frombuffer(self._bufmin, dtype = np.float64)
        self._bufmax = (ct.c_double * (value >> 3))()
        self.max = np.frombuffer(self._bufmax, dtype = np.float64)
        self._npt = value
        return value    
    
    @property 
    def enab(self) -> bool:
        return self._enab
    @enab.setter
    def enab(self, value: bool) -> bool:
        self._enab = value
        dwf.FDwfAnalogInChannelEnableSet(self.hdwf, self._chan, ct.c_bool(value))
        return value 
    
    @property 
    def avg(self) -> bool:
        return self._avg # No check su dispositivo (!)
    @avg.setter
    def avg(self, value: bool) -> bool:
        if value:
            dwf.FDwfAnalogInChannelFilterSet(self.hdwf, self._chan, constants.filterAverage)
        else:
            dwf.FDwfAnalogInChannelFilterSet(self.hdwf, self._chan, constants.filterDecimate)
        self._avg = value    
        return value
    
class TimeVector:
    
    def __init__(self, hdwf: ct.c_int) -> None:
        """
        Classe riservata...
        """
        self.hdwf = hdwf
        self.vals = []
    
    def update_fs(self, value: float) -> None:
        self.dt = 1/value
        if len(self.vals)>0:
            i0 = len(self.vals)/2
            self.vals = np.array([self.dt*(i-i0) for i in range(len(self.vals))])
            
    def update_npt(self,value: int) -> None:
        i0 = value/2
        self.vals = np.array([self.dt*(i-i0) for i in range(value)])
        
    def get_timestamp(self) -> datetime.datetime:
        secUtc = ct.c_uint()
        tick = ct.c_uint()
        tickPerSecond = ct.c_uint()
        dwf.FDwfAnalogInStatusTime(self.hdwf,ct.byref(secUtc),
                                   ct.byref(tick),ct.byref(tickPerSecond))
        self.t0 = datetime.datetime.fromtimestamp(secUtc.value)
    
class Scope:

    def __init__(self, hdwf: ct.c_int, fs: float = 100e6, npt: int = 8192) -> None:
        """
        Inizializza l'oscilloscopio analogico. Fare riferimento ai valori di 
        default indicati a seguire. 

        Parameters
        ----------
        hdwf : ct.c_int
            handle all'Analog Discovery 2 connesso.
        fs : float
            frequenza di sampling in Sa/s.
            [default 100MSa/s]
        npt : int
            numero di punti di acquisizione.
            [default 8192]

        Returns
        -------
        None

        Notes
        -----
        I parameri `fs` e `npt` possono essere 
        aggiornati in seguito, per esempio con 
        
        >>> mioad2 = tdwf.AD()
        >>> # Qui viene usato il default fs = 100MSa/s
        >>> scop = tdwf.Scope(mioad2.hdwf)
        >>> # Qui cambio idea e imposto fs = 100kSa/s
        >>> scop.fs = 1e5
                
        Si  noti che AD2 ammette solo frequenze di campionamento che sono 
        frazioni di 100MSa/s. Gli altri valori verranno arrotondati al valore 
        ammesso più vicino. Per esempio se chiedo 60e6, il valore impostato sarà
        50e6. Se chiedo 40e6, il valore impostato sarà 33.333...e6
        """
        self.hdwf = hdwf
        # Verifica dimensione buffer
        temp1 = ct.c_int()
        dwf.FDwfAnalogInBufferSizeInfo(self.hdwf, 0, ct.byref(temp1))
        self._nmax = temp1.value
        # Inizializza canali
        self.ch1 = ChanIn(hdwf, tdwf.chanCh1)
        self.ch2 = ChanIn(hdwf, tdwf.chanCh2)
        self.time = TimeVector(hdwf)
        self.fs = fs
        self.npt = npt # => anche importante per settare i buffer di ch1/ch2
        
    @property 
    def fs(self) -> float:
        return self._fs # No check su dispositivo (!)
    @fs.setter
    def fs(self,value: float) -> float:
        dwf.FDwfAnalogInFrequencySet(self.hdwf, ct.c_double(value)) 
        temp = ct.c_double()
        dwf.FDwfAnalogInFrequencyGet(self.hdwf, ct.byref(temp)) 
        self._fs = temp.value
        self.time.update_fs(temp.value)
        return value
    
    @property 
    def npt(self) -> int:
        return self._npt # No check su dispositivo (!)
    @npt.setter
    def npt(self, value: int) -> int:
        value = max(1,value)
        if value <= self._nmax:
            # Acquisizione singola
            dwf.FDwfAnalogInBufferSizeSet(self.hdwf, ct.c_int(int(value)))
            dwf.FDwfAnalogInNoiseSizeSet(self.hdwf, ct.c_int(int(value) >> 3))
            dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf, constants.acqmodeSingle)
            #print("Single acquisition mode")
        else:
            # Aquisizione "record", non raccomandato > 1MSa/s
            dwf.FDwfAnalogInBufferSizeSet(self.hdwf, ct.c_int(self._nmax))
            dwf.FDwfAnalogInNoiseSizeSet(self.hdwf, ct.c_int(self._nmax >> 3))
            dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf, constants.acqmodeRecord) 
            #print("Record mode")
        self.ch1.npt = value
        self.ch2.npt = value
        self.time.update_npt(value)
        self._npt = value    
        return value    
        
    @property
    def status(self) -> ct.c_ubyte:
        tmp = ct.c_ubyte()  
        dwf.FDwfAnalogInStatus(self.hdwf, ct.c_bool(True), ct.byref(tmp))
        return tmp
    
    def trig(self, enable: bool, timeout: float = 1.0, level: float = 0.0, 
             sour = tdwf.trigsrcCh1, hist: float = 0.01, holdoff: float = 0.0, 
             delay: float = 0.0, cond = tdwf.trigslopeRise) -> None:
        """
        Imposta il trigger per l'oscilloscopio analogico. 
        Notare: quasi tutti i parametri sono fissati per default e spesso
        non serve indicarli. Di default, se non viene chiamata questo funzione
        il trigger è disattivato.

        Parameters
        ----------
            enable : bool
                Attiva (se `True`) il trigger.
            timeout: float
                Quanto aspetta se il trigger non scatta.
                [default 1.0s]
            level : float
                Livello di trigger.
                [default 0.0V]
            sour : ct.c_int
                Canale di trigger.
                [default `tdwf.trigsrcCh1`]
                Altre opzioni: `tdwf.trigsrcCh2`, `tdwf.trigsrcW1`, `tdwf.trigsrcW2`,
                `tdwf.trigsrcT1`, `tdwf.trigsrcT2`.
            hist : float
                Isteresi di trigger.
                [default 0.01V]
            holdoff : float
                Pausa prima di armare il trigger.
                [default 0.0s]
            delay : float
                Ritardo trigger rispetto al tempo zero (centro acquisizione).
                [default 0.0s]
            cond : ct.c_int
                Condizione di trigger.
                [default `tdwf.trigslopeRise`]
                Altre opzini: `trigslopeFall` e `trigslopeEither`.

        Returns
        -------
        None

        Examples
        --------

        >>> mioad2 = twdf.AD()
        >>> oscillo = twdf.Scope(mioad2.hdwf)
        >>> # Attiva trigger su Ch1 con livello 0V
        >>> oscillo.trig(True) 
        >>> # Trigger su Ch2 con livello 1V 
        >>> oscillo.trig(True, level=1, sour=tdwf.trigsrcCh2)

        """
        if enable:
            if sour == tdwf.trigsrcCh1:
                dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, tdwf.trigsrcAIn) 
                dwf.FDwfAnalogInTriggerChannelSet(self.hdwf, tdwf.chanCh1)
            elif sour == tdwf.trigsrcCh2:
                dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, tdwf.trigsrcAIn) 
                dwf.FDwfAnalogInTriggerChannelSet(self.hdwf, tdwf.chanCh2)
            else:
                dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, sour) 
                dwf.FDwfAnalogInTriggerChannelSet(self.hdwf, ct.c_ubyte(0)) 
            dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, ct.c_double(timeout)) 
            dwf.FDwfAnalogInTriggerTypeSet(self.hdwf, tdwf.trigtypeEdge)
            dwf.FDwfAnalogInTriggerLevelSet(self.hdwf, ct.c_double(level)) 
            dwf.FDwfAnalogInTriggerHysteresisSet(self.hdwf, ct.c_double(hist)) 
            dwf.FDwfAnalogInTriggerHoldOffSet(self.hdwf, ct.c_double(holdoff))
            dwf.FDwfAnalogInTriggerConditionSet(self.hdwf, cond) 
            # Position
            #xpt = min(max(0,delay)*self.fs,self.npt) - self.npt/2
            #pos = -xpt/self.fs
            #dwf.FDwfAnalogInTriggerPositionSet(self.hdwf, ct.c_double(pos))
            dwf.FDwfAnalogInTriggerPositionSet(self.hdwf, ct.c_double(delay))
        else:
            dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, ct.c_double(0)) 
       
    def sampleXL(self) -> None:
        """
        Avvia una acquisizione che va oltre i limiti di self.sample. Vedere la 
        documentazione di self.sample per ulteriori dettagli.
        """
        cAvai = ct.c_int()
        cLost = ct.c_int()
        cCorr = ct.c_int()
        dwf.FDwfAnalogInRecordLengthSet(self.hdwf, ct.c_double(self._npt/self.fs))
        # Avvio registrazione
        dwf.FDwfAnalogInConfigure(self.hdwf, ct.c_bool(False), ct.c_bool(True))
        # Loop di registrazione
        ii = 0
        while ii < self._npt:
            #print(ii, self._device.error)
            stv = self.status.value
            if ii == 0 and (stv == constants.DwfStateConfig.value or 
                            stv == constants.DwfStatePrefill.value or 
                            stv == constants.DwfStateArmed.value) :
                continue # Acquisizione non ancora partita
        
            dwf.FDwfAnalogInStatusRecord(self.hdwf, ct.byref(cAvai), ct.byref(cLost), ct.byref(cCorr))
            ii += cLost.value + cCorr.value
            if cAvai.value==0 :
                continue
            if cAvai.value > self._npt - ii:
                cAvai = ct.c_int(self._npt - ii)
            
            if self.ch1._enab:
                ptr = ct.byref(self.ch1._buffer, ct.sizeof(ct.c_double)*ii) 
                dwf.FDwfAnalogInStatusData(self.hdwf, tdwf.chanCh1, ptr, cAvai) 
            if self.ch2._enab:
                ptr = ct.byref(self.ch2._buffer, ct.sizeof(ct.c_double)*ii) 
                dwf.FDwfAnalogInStatusData(self.hdwf, tdwf.chanCh2, ptr, cAvai) 
            ii += cAvai.value
    
    def sample16XL(self) -> None:
        """
        Avvia una acquisizione che va oltre i limiti di self.sample. Vedere la 
        documentazione di self.sample per ulteriori dettagli. Versione raw.
        """
        cAvai = ct.c_int()
        cLost = ct.c_int()
        cCorr = ct.c_int()
        dwf.FDwfAnalogInRecordLengthSet(self.hdwf, ct.c_double(self._npt/self.fs))
        # Avvio registrazione
        dwf.FDwfAnalogInConfigure(self.hdwf, ct.c_bool(False), ct.c_bool(True))
        # Loop di registrazione (raw)
        ii = 0
        while ii < self._npt:
            #print(ii, self._device.error)
            stv = self.status.value
            if ii == 0 and (stv == constants.DwfStateConfig.value or 
                            stv == constants.DwfStatePrefill.value or 
                            stv == constants.DwfStateArmed.value) :
                continue # Acquisizione non ancora partita
        
            dwf.FDwfAnalogInStatusRecord(self.hdwf, ct.byref(cAvai), ct.byref(cLost), ct.byref(cCorr))
            ii += cLost.value + cCorr.value
            if cAvai.value==0 :
                continue
            if cAvai.value > self._npt - ii:
                cAvai = ct.c_int(self._npt - ii)
            
            if self.ch1._enab:
                ptr = ct.byref(self.ch1._buffer16, ct.sizeof(ct.c_int16)*ii) 
                dwf.FDwfAnalogInStatusData16(self.hdwf, tdwf.chanCh1, ptr, ct.c_int(0), cAvai) 
            if self.ch2._enab:
                ptr = ct.byref(self.ch2._buffer16, ct.sizeof(ct.c_int16)*ii) 
                dwf.FDwfAnalogInStatusData16(self.hdwf, tdwf.chanCh2, ptr, ct.c_int(0), cAvai) 
            ii += cAvai.value
        # Conversione finale a double
        if self.ch1._enab:
            self.ch1.vals = self.ch1.vals16*self.ch1.rng/65536 + self.ch1.offs
        if self.ch2._enab:
            self.ch2.vals = self.ch2.vals16*self.ch2.rng/65536 + self.ch2.offs
            
    def sample(self) -> None:
        """
        Avvia una acquisizione ADC in base ai parametri impostati. Se il numero 
        di punti è superiore al massimo buffer, chiama automaticamente la versione
        sampleXL, che però funziona solo a sampling rate ridotto (indicativamente
        fino a 1MSa/s).

        Returns
        -------
        None

        Notes
        -----

        * La funzione è bloccante e continua solo a fine acquisizione
        * La funziona non ritorna alcun risultato. I risultati sono disponibili nei
        vettori

            * `[Scope].time.vals` => vettore dei tempi
            * `[Scope].time.t0` => timestamp del trigger
            * `[Scope].ch1.vals` => voltaggi misurati
            * `[Scope].ch1.min` => vettore dei valori minimi (numero in rapporto 1:4 con misura)
            * `[Scope].ch1.max` => vettore dei valori massimi (numero in rapporto 1:4 con misura)

        """
        if self._npt > self._nmax:
            self.sampleXL()
        else:
            dwf.FDwfAnalogInConfigure(self.hdwf, ct.c_bool(False), ct.c_bool(True))
            while not self.status.value == constants.DwfStateDone.value:
                time.sleep(0.001)
            if self.ch1._enab:
                dwf.FDwfAnalogInStatusData(self.hdwf, tdwf.chanCh1, self.ch1._buffer, ct.c_int(self._npt))
                dwf.FDwfAnalogInStatusNoise(self.hdwf, tdwf.chanCh1, 
                    self.ch1._bufmin, self.ch1._bufmax, ct.c_int(self._npt >> 3))
            if self.ch2._enab:
                dwf.FDwfAnalogInStatusData(self.hdwf, tdwf.chanCh2, self.ch2._buffer, ct.c_int(self._npt))
                dwf.FDwfAnalogInStatusNoise(self.hdwf, tdwf.chanCh2, 
                    self.ch2._bufmin, self.ch2._bufmax, ct.c_int(self._npt >> 3))
            self.time.get_timestamp()            
                
    def sample16(self) -> None:
        """
        Avvia una acquisizione ADC in base ai parametri impostati. Se il numero 
        di punti è superiore al massimo buffer, chiama automaticamente la versione
        sampleXL, che però funziona solo a sampling rate ridotto (indicativamente
        fino a 1MSa/s). Versione raw.

        Returns
        -------
        None

        Notes
        -----
        
        * La funzione è bloccante e continua solo a fine acquisizione
        * La funziona non ritorna alcun risultato. I risultati sono disponibili nei
          vettori

            * `[Scope].time.vals` => vettore dei tempi
            * `[Scope].time.t0` => timestamp del trigger
            * `[Scope].ch1.vals` => voltaggi misurati
            * `[Scope].ch1.min` => vettore dei valori minimi (numero in rapporto 1:4 con misura)
            * `[Scope].ch1.max` => vettore dei valori massimi (numero in rapporto 1:4 con misura)

        """
        if self._npt > self._nmax:
            self.sample16XL()
        else:
            dwf.FDwfAnalogInConfigure(self.hdwf, ct.c_bool(False), ct.c_bool(True))
            while not self.status.value == constants.DwfStateDone.value:
                time.sleep(0.001)
                if self.ch1._enab:
                    dwf.FDwfAnalogInStatusData16(self.hdwf, tdwf.chanCh1, 
                        self.ch1._buffer16, ct.c_int(0), ct.c_int(self._npt))
                    dwf.FDwfAnalogInStatusNoise(self.hdwf, tdwf.chanCh1, 
                        self.ch1._bufmin, self.ch1._bufmax, ct.c_int(self._npt >> 3))
                    self.ch1.vals = self.ch1.vals16*self.ch1.rng/65536 + self.ch1.offs
                if self.ch2._enab:
                    dwf.FDwfAnalogInStatusData16(self.hdwf, tdwf.chanCh2, 
                        self.ch2._buffer16, ct.c_int(0), ct.c_int(self._npt))
                    dwf.FDwfAnalogInStatusNoise(self.hdwf, tdwf.chanCh2, 
                        self.ch2._bufmin, self.ch2._bufmax, ct.c_int(self._npt >> 3))
                    self.ch2.vals = self.ch2.vals16*self.ch2.rng/65536 + self.ch2.offs
                self.time.get_timestamp()
        
    def reset(self) -> None:
        """
        Resetta l'oscilloscopio.
        """
        dwf.FDwfAnalogInReset(self.hdwf) 

