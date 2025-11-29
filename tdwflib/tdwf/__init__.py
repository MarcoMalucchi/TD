# -*- coding: utf-8 -*-
"""
    `_T_D_W___ ----> TecnoDig Wrapper`      
    `___D_W_F_ ----> Digilent WaveForms`

Questo è un "wrapper" python semplificato per la SDK Digilent versione 3.22. 
Il dispositivo target è 
 
     `Analog Discovery 2`
     
e il codice non è stato testato con altre versioni della SDK e/o con diversi 
dispositivo Digilent, quindi non c'è nessuna garanzia che funzioni se uscite
dall'ambito per cui è stato creato.

Note aggiuntive / disclaimer:    

* il codice NON ambisce ad essere un wrapper generico Digilent, fa quello che fa,
  così come può
* non c'è NESSUN MANUALE, solo alcuni semplici esempi e un po' di documentazione
  inclusa nel modulo stesso
* il codice è disegnato per permettere una interazione il più possibile
  semplice e diretta con l'hardware, ma (non si può avere tutto) fa ben pochi
  controlli sui possibili errori
* TUTTAVIA il codice espone alcune funzioni di base per controllare "a mano"
  eventuali codici e messaggi di errore comunicati dal dispositivo, e in ogni
  caso il codice dovrebbe (!) essere robusto per l'uso "standard" (*whatever
  that means...*)

Note implementative / curiosità:

* il grosso della (relativa...) "cripticità" del codice deriva dall'uso 
  sistematico della libreria `ctypes`
* `ctypes` è usata perché la SDK richiede per argomnti delle variabili con 
  una precisa rappresentazione binaria in memoria. Per esempio `uint32`... 
  un intero senza segno da 32 bit. In python `int` è qualcosa si molto meno 
  ben definito ed è necessario "tradurre" queste variabili con `ctypes`.
* un'altra complessità deriva dalla necessità di avere un wrapper che 
  funzioni su Windows, MacOS e linux (ossia *cross-platform*). Qui sono stati
  banalmente copiati gli esempi e campioni forniti da Digilent.

Issues:
- manca implementazione driver SPI 
- manca implementazione Logic Analyzer

Autori: Stefano Roddaro

Last Update: 240818
"""
from .device import *
from .analogin import *
from .analogout import *
from .digital import *
from .protocols import *
from .tdwfconstants import *
