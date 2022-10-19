# Enaho`
import requests
import os
import time
from os.path import join
from os import listdir, rmdir
from shutil import move
from zipfile import ZipFile, BadZipfile

PeriodosModulo = {
    2018: '634',
    2019: '687',
    2020: '737'}

fechas = [2018,2020]
Encuestas = [PeriodosModulo[per] for per in PeriodosModulo.keys() if (per>=fechas[0]) & (per<=fechas[1])]
PeriodosModulo[2018]

















