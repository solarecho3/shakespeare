import os
import logging
import yaml
from typing import Any

class DataLoader:

    def __init__(self, datapath: str | os.PathLike="../data/"):
        """
        A class to load data.

        Parameters
        ----------

        path: str | os.PathLike - Relative path to the data folder. Default = ../data/
        """
        
        self.configpath = os.path.abspath('../config/')
        
        with open(
            f'{self.configpath}/datasets.yml',
            'r',
            encoding='utf-8') as f:

            self.configs = yaml.load(f, Loader=yaml.Loader)

    def tiny(self):
        """
        Load the tiny shakespeare data set.
        """
        
        with open(
            self.configs['tiny']['path'],
            'r',
            encoding='utf-8') as f:

            text = f.read()
        
        self.data = text

    def complete(self):
        """
        Load The Complete Works of William Shakespeare by Gutenberg data set.
        """
        with open(
            self.configs['complete']['path'],
            'r',
            encoding='utf-8') as f:

            text = f.read()

        self.data = text

    