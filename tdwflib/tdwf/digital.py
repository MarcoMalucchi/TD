# -*- coding: utf-8 -*-
"""
[tdwf/digital.py]
Controllo della parte digitale di Analog Discovery 2.
Nota: l'oscilloscopio digitale (logic analyzer) non è stato implementato
"""

import ctypes as ct        
from .config import dwf, constants    
import tdwf.tdwfconstants as tdwf   

class DIO:
    
    # Si assume sia un canale di tipo pulse con push-pull
    
    def __init__(self, hdwf: ct.c_int, _chan: ct.c_int) -> None:
        self.hdwf = hdwf
        self._chan = _chan
        self.pwm_setup(enab = False)   
        
    def pwm_setup(self, nLH: list[int] = [50000, 50000], n0: int = 1,
                   div: int = 1, enab: bool = True) -> None:
        """
        Configura il canale output digitale specifico.
       
        Patemeters:
        nLH : List[int]
            Numero di conteggi bassi e alti nel pattern
            [default [50000,50000])
        n0 : int
            Numero di conteggi di delay iniziale 
            [default 1]
        div : int
            Divisore rispetto a clock da 100MHz
            [default 1]
        enab : bool
            Abilitazione del canale 
            [default True]
            
        Returns
        -------
        None

        Notes
        -----

        * La configurazione di default corrisponde a un clock di 1kHz e duty al 50%.
        * Non è mai possibile impostare zero conteggi per `nHL` o `n0`, il minimo è sempre 1. 
        * Non sono stati implementati alcuni dettagli della SDK che sono stati giudicati poco cruciali, 
        come il fatto che la riproduzione parta da valore logico alto o basso, oppure il numero di 
        ripetizioni del pattern (sempre infinito qui).

        Examples
        --------

        Esempio di configurazione di D01 e D02 con due onde a 40kHz (2500 conteggi totali di clock)
        e sfasate di 90 gradi (`n0` aumentato nel secondo caso di 625 conteggi)

        >>> logic = tdwf.MultiDIO(ad2.hdwf)
        >>> logic.dio[1].pwm_setup(nHL = [1250, 1250])
        >>> logic.dio[2].pwm_setup(nHL = [1250, 1250], n0=625+1)
        >>> logic.pwm_start()

        Si noti che le porte digitali di contano da 0.
        """
        self.nLH = nLH
        self.n0 = n0
        self.div = div
        self.enab = enab
    
    @property
    def nLH(self) -> list[int]:
        return self._nLH
    @nLH.setter
    def nLH(self, value: list[int]) -> list[int]:
        dwf.FDwfDigitalOutCounterSet(self.hdwf, self._chan, ct.c_uint(value[0]), ct.c_uint(value[1]))
        self._nLH = value
        return value  
    
    @property
    def n0(self) -> int:
        return self._n0
    @n0.setter
    def n0(self, value: int) -> int:
        # Inizio fisso low
        starthigh = ct.c_bool(False)
        dwf.FDwfDigitalOutCounterInitSet(self.hdwf, self._chan, starthigh, ct.c_uint(value))
        self._n0 = value
        return value  
    
    @property
    def div(self) -> int:
        return self._div
    @div.setter
    def div(self, value: int) -> int:
        dwf.FDwfDigitalOutDividerSet(self.hdwf, self._chan, ct.c_uint(value))
        self._div = value
        return value  
    
    @property
    def enab(self) -> bool:
        return self._enable
    @enab.setter
    def enab(self, value: bool) -> bool:
        dwf.FDwfDigitalOutEnableSet(self.hdwf, self._chan, ct.c_bool(value))
        self._enab = value
        return value 
        
class MultiDIO:
    
    def __init__(self, hdwf: ct.c_int) -> None:
        """
        Configurazione del digital input/output mulitplo.
        Sostanzialmente è solo un "contenitore". 
        
        Notes
        -----
        Al momento non l'oscilloscopio digitale non è implementato.
        """
        self.hdwf = hdwf
        self.dio = [ DIO(hdwf, ct.c_int(ii)) for ii in range(16) ]
        dwf.FDwfDigitalInSampleFormatSet(self.hdwf,ct.c_int(16))
    
    def pwm_start(self) -> None:
        """
        Attiva il pattern generator in base alla configurazione dei vari canali
        digitali abilitati e configurati.
        """
        dwf.FDwfDigitalOutConfigure(self.hdwf, ct.c_bool(True)) 
    
    def pwm_stop(self) -> None:
        """
        Ferma il pattern generator.
        """
        dwf.FDwfDigitalOutConfigure(self.hdwf, ct.c_bool(False)) 
    
    def reset(self) -> None:
        """
        Resetta la parte logica di Analog Discovery 2.
        """
        dwf.FDwfDigitalInReset(self.hdwf) 
        dwf.FDwfDigitalOutReset(self.hdwf) 
        dwf.FDwfDigitalIOReset(self.hdwf) 
        # reset di tutto...

    def outputmask(self, value: int) -> None:
        """
        Attiva le linee di output in base al valore binario dell'argomento
        specificato. La configurazione non include le linee configurate 
        come PWM. 

        Parameters
        ----------
        value : int
            Maschera di 16 bit per l'attivazione delle linee digitali.

        Returns
        -------
        None

        Notes
        -----
        Esistono varie costanti utili come `tdwf.D00`, `tdwf.D01` e così via. Le seguenti
        istruzioni sono tutte equivalenti e attivano D00 e D04

        >>> logic = tdwf.MultiDIO(ad2.hdwf)
        >>> # Usando le costanti e un or, ossia |
        >>> logic.outputmask(tdwf.D00 | tdwf.D04)
        >>> # Binario          FEDCBA9876543210
        >>> logic.outputmask(0b0000000000010001)
        >>> # Esadecimale
        >>> logic.outputmask(0x0011)
        >>> # Decimale
        >>> logic.outputmask(9)
        """
        dwf.FDwfDigitalIOOutputEnableSet(self.hdwf, ct.c_uint(value))
        
    def write(self, value: int) -> None:
        """
        Scrive i canali di output digitali.

        Parameters
        ----------
        value : int
            Campo di 16 bit per i valori di output sui 16 canali

        Returns
        -------
        None

        Notes
        -----

        Esistono varie costanti utili come `tdwf.D00`, `tdwf.D01` e così via. Le seguenti
        istruzioni sono tutte equivalenti e alzano il valore logico di D00 e D04

        >>> logic = tdwf.MultiDIO(ad2.hdwf)
        >>> # Usando le costanti e un or, ossia |
        >>> logic.write(tdwf.D00 | tdwf.D04)
        >>> # Binario     FEDCBA9876543210
        >>> logic.write(0b0000000000010001)
        >>> # Esadecimale
        >>> logic.write(0x0011)
        >>> # Decimale
        >>> logic.write(9)
        """
        dwf.FDwfDigitalIOOutputSet(self.hdwf, ct.c_uint(value))
    
    def read(self) -> int:
        """
        Legge tutte le linee di input attive

        Returns
        -------
        int
            Numero uint16, con un bit per ogni porta logica

        Notes
        -----

        Esistono varie costanti utili come `tdwf.D00`, `tdwf.D01` e così via. Queste
        possono essere usate per estrarre i valori booleani corrispondenti

        >>> logic = tdwf.MultiDIO(ad2.hdwf)
        >>> val = logic.read()
        >>> # Verifichiamo se D03 è True o False
        >>> print(f"La porta D03 è {(val&tdwf.D03 > 0)}")  

        Dove abbiamo sfruttato la maschera di bit di `tdwf.D03` che corrisponde a `0x0008`
        o `0b0000000000001000`, con un solo `1` nella posizione corretta.
        """
        dwf.FDwfDigitalIOStatus(self.hdwf)
        temp = ct.c_uint()
        dwf.FDwfDigitalIOInputStatus(self.hdwf, ct.byref(temp))
        return temp.value