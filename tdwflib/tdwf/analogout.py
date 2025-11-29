# -*- coding: utf-8 -*-
"""
Controllo generatore di funzioni
--------------------------------

Analog Discovery 2 contiene due DAC e la SDK espone vari metodi per il loro 
controllo. Le forme d'onda sono definite tramite tre "nodi":
    
1 **Carrier**. Controlla la forma d'onda
2. **AM Modulation**. Ulteriore forma d'onda (in variazione % di ampiezza) che controlla la modulazione di ampiezza.
3. **FM Modulation**. Ulteriore forma d'onda (in variazione % di velocità di fase) che controlla la modulazione di frequenza.
"""

import ctypes as ct        
from .config import dwf, constants    
import tdwf.tdwfconstants as tdwf  
     
class ChanOut:
    
    def __init__(self, hdwf: ct.c_int, _nmax: int, _chan: ct.c_int) -> None:
        """
        Classe riservata... 
        """
        self._chan = _chan
        self.hdwf = hdwf
        self.carrier = NodeOut(hdwf, _nmax, _chan, tdwf.nodeCA)
        self.freqmod = NodeOut(hdwf, _nmax, _chan, tdwf.nodeFM)
        self.amplmod = NodeOut(hdwf, _nmax, _chan, tdwf.nodeAM)
        self.carrier.config()

    def sync(self) -> None:
        """
        Sincronizza il canale output con l'altro canale, che diventa il 
        "master". Dopo l'esecuzione il funzionamento di entrambi i canali 
        è controllato da "master".
        
        Returns 
        -------
        None

        Notes
        -----
        L'impostazione viene "cancellata" se viene richiesto direttamente lo start
        dello stesso canale.

        >>> wgen.w1.sync()   # sincronizza w1 a w2
        >>> wgen.w1.start()  # canella impostazione e avvia w1 direttamente

        Examples
        --------
        Esempio di avvio sincronizzato: il controllo del canale `w1 ` viene "ceduto"
        a `w2`, che controlla entrambi. La relazione di fase fra i canali è fissata
        e se (!) la frequenza è uguale avranno la stessa fase.

        >>> wgen.w1.sync()   
        >>> wgen.w2.start()  

        Esempio di avvio NON sincronizzato (relazione di fase random)

        >>> wgen.w1.start() 
        >>> wgen.w2.start() 
        """
        if self._chan == tdwf.chanW1:
            dwf.FDwfAnalogOutMasterSet(self.hdwf, tdwf.chanW1, tdwf.chanW2)
        else:
            dwf.FDwfAnalogOutMasterSet(self.hdwf, tdwf.chanW2, tdwf.chanW1)
            
    def start(self) -> None:
        """
        Avvia il canale analog out specifico, in base alla configurazione 
        attuale. 

        Returns 
        -------
        None

        Notes
        -----
        Il comportamento del geneatore dipende dal parametro `autoconf`,
        impostato in fase di avvio di Analog Discovery 2
         * `tdwf.AD(autoconf=3)`, che è default, impone che il generatore 
         aggionra "on the fly" le proprietà dell'onda. Per esempio, se `w1`
         è in riproduzione e viene cambiata `w1.ampl`, il cambiamento sarà
         immediato.
         * per valori diversi di `autoconf`, cambiare un parametro 
         *fermerà la rirpoduzione* e sarà necessario far ripartire il 
         generatore con una nuova istruzione di start.
        """
        dwf.FDwfAnalogOutConfigure(self.hdwf, self._chan, ct.c_bool(True))
        
    def stop(self) -> None:
        """
        Ferma la riproduzione sul canale analog out specifico.
        """
        dwf.FDwfAnalogOutConfigure(self.hdwf, self._chan, ct.c_bool(False))
    
    # Seguono vari "forward" per mascherare la presenza dei tre nodi...

    def config(self, freq: float = 1000.0, offs: float = 0.0, ampl: float = 1.0, 
               duty: float = 50, phi: float = 0.0, func: ct.c_int = tdwf.funcSine, 
               data: list[float] = [], enab: bool = True) -> None:
        """
        Configura il canale output analogico specifico. 
            
        Patemeters
        ----------
        freq : float
            Frequenza di riproduzione. 
            [default 1kHz]
        offs : float
            Offset. 
            [default 0.0V]
        ampl : float
            Ampiezza di picco. 
            [default 1.0V]
        duty : float
            Duty cycle in %. 
            [default 50%]
        phi : float
            Sfasamento in gradi 
            [default 0.0deg]
        func : ct.c_int
            Tipo di forma d'onda. 
            [default `tdwf.funcSine`]
            Altre opzioni standard: `tdwf.funcDC`, `tdwf.funcSquare`,
            `tdwf.funcTriangular`, `tdwf.funcCustom`.
        data : List[float]
            Forma d'onda generica.
            [default lista nulla]
            Ricordare che i valori devono stare entro +/- 1.
        enab : bool
            Abilitazione del canale 
            [default `True`]
            
        Returns
        -------
        None

        Notes
        -----
        Si noti che i singoli parametri possono essere anche impostati individualmente, per esempio

        >>> ad2 = tdwf.AD()
        >>> wgen = tdwf.WaveGen(ad2.hdwf)
        >>> wgen.w1.ampl = 1.234
        >>> wgen.w2.freq = 5678.9

        L'impostazione di una forma d'onda custom richiede di modificare la proprietà `data` e di 
        scegliere il corretto valore di `func`

        >>> wgen.w1.config(data=[0, 0.2, 0.4, 1.0], func=tdwf.funcCustom)
        >>> # o alternativamente...
        >>> wgen.w1.data = [... vettore valori...]
        >>> wgen.w1.func = tdwf.funcCustom
        >>> wgen.w1.start()

        Si ricorda che i valori di `data` devono essere compresi fra -1 e +1. I valori fuori range 
        verranno clippati automaticamente. I valori devono intendersi in unità di `ampl`. Ossia per 
        esempio se l'ampiezza specificata da `ampl` è 2.5, allora nei punti in cui `data` vale 1 il 
        canale genererà 2.5V.
        """
        self.carrier.config(freq, offs, ampl, duty, phi, func, data, enab)
        
    @property
    def ampl(self) -> float:
        return self.carrier.ampl
    @ampl.setter
    def ampl(self, value: float) -> float:
        self.carrier.ampl = value
        return value
    
    @property
    def offs(self) -> float:
        return self.carrier.offs
    @offs.setter
    def offs(self, value: float) -> float:
        self.carrier.offs = value
        return value      
    
    @property
    def freq(self) -> float:
        return self.carrier.freq
    @freq.setter
    def freq(self, value: float) -> float:    
        self.carrier.freq = value
        return value  
    
    @property
    def phi(self) -> float:
        return self.carrier.phi
    @phi.setter
    def phi(self, value: float) -> float:  
        self.carrier.phi = value
        return value  
    
    @property
    def func(self) -> ct.c_int:
        return self.carrier.func
    @func.setter
    def func(self, value: ct.c_int):
        self.carrier.func = value
        return value  
    
    @property
    def duty(self) -> float:
        return self.carrier.duty
    @duty.setter
    def duty(self, value: float) -> float:
        self.carrier.duty = value
        return value  
    
    @property
    def data(self) -> list[float]:
        return self.carrier.data
    @data.setter
    def data(self, value: list[float]) -> list[float]:
        self.carrier.data = value
        return value  
    
    @property
    def enab(self) -> bool: 
        return self.carrier.enab
    @enab.setter
    def enab(self, value: bool) -> bool:
        self.carrier.enab = value
        return value  
        
class NodeOut:
    
    def __init__(self, hdwf: ct.c_int, _nmax: int, _chan: ct.c_int,
                  _node: ct.c_int) -> None:
        """
        Classe riservata... 
        """
        self.hdwf = hdwf
        self._nmax = _nmax
        self._chan = _chan
        self._node = _node
        self._buffer = (ct.c_double * _nmax)()
        
    def config(self, freq: float = 1000.0, offs: float = 0.0, ampl: float = 1.0, 
               duty: float = 50, phi: float = 0.0, func: ct.c_int = tdwf.funcSine, 
               data: list[float] = [], enab: bool = True) -> None:
        """
        Configura il canale output analogico specifico. 
            
        Patemeters
        ----------
        freq : float
            Frequenza di riproduzione. 
            [default 1kHz]
        offs : float
            Offset. 
            [default 0.0V]
        ampl : float
            Ampiezza di picco. 
            [default 1.0V]
        duty : float
            Duty cycle in %. 
            [default 50%]
        phi : float
            Sfasamento in gradi 
            [default 0.0deg]
        func : ct.c_int
            Tipo di forma d'onda. 
            [default `tdwf.funcSine`]
            Altre opzioni standard: `tdwf.funcDC`, `tdwf.funcSquare`,
            `tdwf.funcTriangular`, `tdwf.funcCustom`.
        data : List[float]
            Forma d'onda generica.
            [default lista nulla]
            Ricordare che i valori devono stare entro +/- 1.
        enab : bool
            Abilitazione del canale 
            [default `True`]
            
        Returns
        -------
        None

        Notes
        -----
        Si noti che i singoli parametri possono essere anche impostati individualmente, per esempio

        >>> ad2 = tdwf.AD()
        >>> wgen = tdwf.WaveGen(ad2.hdwf)
        >>> wgen.w1.ampl = 1.234
        >>> wgen.w2.freq = 5678.9

        L'impostazione di una forma d'onda custom richiede di modificare la proprietà `data` e di 
        scegliere il corretto valore di `func`

        >>> wgen.w1.config(data=[0, 0.2, 0.4, 1.0], func=tdwf.funcCustom)
        >>> # o alternativamente...
        >>> wgen.w1.data = [... vettore valori...]
        >>> wgen.w1.func = tdwf.funcCustom
        >>> wgen.w1.start()

        Si ricorda che i valori di `data` devono essere compresi fra -1 e +1. I valori fuori range 
        verranno clippati automaticamente. I valori devono intendersi in unità di `ampl`. Ossia per 
        esempio se l'ampiezza specificata da `ampl` è 2.5, allora nei punti in cui `data` vale 1 il 
        canale genererà 2.5V.
        """
        self.ampl = ampl
        self.offs = offs
        self.freq = freq
        self.phi  = phi
        self.duty = duty
        self.func = func
        if len(data)>0:
            self.data = data
        self.enab = enab
        
    @property
    def ampl(self) -> float:
        return self._ampl
    @ampl.setter
    def ampl(self, value: float) -> float:
        dwf.FDwfAnalogOutNodeAmplitudeSet(self.hdwf, self._chan, self._node, ct.c_double(value))
        self._ampl = value
        return value
    
    @property
    def offs(self) -> float:
        return self._offs
    @offs.setter
    def offs(self, value: float) -> float:
        dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf, self._chan, self._node, ct.c_double(value))
        self._offs = value
        return value      
    
    @property
    def freq(self) -> float:
        return self._freq
    @freq.setter
    def freq(self, value: float) -> float:    
        dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, self._chan, self._node, ct.c_double(value))
        self._freq = value
        return value  
    
    @property
    def phi(self) -> float:
        return self._phi
    @phi.setter
    def phi(self, value: float) -> float:  
        dwf.FDwfAnalogOutNodePhaseSet(self.hdwf, self._chan, self._node, ct.c_double(value))
        self._phi = value
        return value  
    
    @property
    def func(self) -> ct.c_int:
        return self._func
    @func.setter
    def func(self, value: ct.c_int) -> ct.c_int:
        dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, self._chan, self._node, value)
        self._func = value
        return value  
    
    @property
    def duty(self) -> float:
        return self._duty
    @duty.setter
    def duty(self, value: float) -> float:
        dwf.FDwfAnalogOutNodeSymmetrySet(self.hdwf, self._chan, self._node, ct.c_double(value))
        self._duty = value
        return value  
    
    @property
    def data(self) -> list[float]:
        return self._data
    @data.setter
    def data(self, value: list[float]) -> list[float]:
        for ii, dd in enumerate(value):
            if ii < self._nmax:
                self._buffer[ii] = dd
        dwf.FDwfAnalogOutNodeDataSet(self.hdwf, self._chan, self._node, self._buffer,
                                     min(len(value), self._nmax))
        self._data = value
        return value  
    
    @property
    def enab(self) -> bool: 
        return self._enab
    @enab.setter
    def enab(self, value: bool) -> bool:
        dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, self._chan, self._node, ct.c_bool(value))
        self._enab = value
        return value  
        
class WaveGen:    
    
    def __init__(self, hdwf: ct.c_int) -> None:
        """
        Inizializza il generatore di funzioni

        Parameters
        ----------
        hdwf : ct.c_int
            Handle al dispositivo Analog Discovery 2 connesso.

        Returns
        -------
        None
        """
        self.hdwf = hdwf
        # Verifica dimensioni buffer
        temp1 = ct.c_int()
        temp2 = ct.c_int()
        dwf.FDwfAnalogOutNodeDataInfo(self.hdwf, ct.c_int(0), ct.c_int(0),
            ct.byref(temp1),ct.byref(temp2))
        self._nmin = temp1.value
        self._nmax = temp2.value
        for inode in range(3):
            dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, tdwf.chanW1, ct.c_int(inode), ct.c_bool(False))
            dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, tdwf.chanW2, ct.c_int(inode), ct.c_bool(False))
        self.w1 = ChanOut(hdwf, self._nmax, tdwf.chanW1)
        self.w2 = ChanOut(hdwf, self._nmax, tdwf.chanW2)
        
    def reset(self) -> None:
        """
        Resetta il generatore di funzioni.
        """
        dwf.FDwfAnalogOutReset(self.hdwf) 
