# -*- coding: utf-8 -*-
"""
[tdwf/setup.py]

Assumiamo che il pacchetto sia nel formto zippato con nome "tdwf.zip"

    pip install tdwf.zip
    
La procedura funziona anche per aggiornare la libreria ad una nuova versione.
Per verificare la corretta installazione del modulo è possibile usare

    pip show tdwf
    
Il comando indicherà la versione e altre informazioni utili, come per esempio
la posizione nel filesystem. Se per caso si volesse disinstallare la libreria,
si può usare

    pip uninstall tdwf
    
"""

from setuptools import setup, find_packages

setup(
    name="tdwf",  
    version="1.1.0",
    author="Stefano Roddaro",
    author_email="stefano.roddaro@unipi.it",
    description="A simplified Python wrapper for the Digilent WaveForms SDK",
    packages=find_packages(),  
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy", "datetime"
    ],
)
