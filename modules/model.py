import os
import logging
"""Accesses data, updates the model."""

class DataLoader:

    def __init__(self, datapath: str | os.PathLike="../data/"):
        """
        A class to load data.

        Parameters
        ----------

        path: str | os.PathLike - Relative path to the data folder. Default = ../data/
        """
        
        self.datapath = os.path.abspath(datapath)
        print(f"DataLoader initialized with {self.datapath}...")

    def tiny(self):
        """
        Load the tiny shakespeare data set.
        """
        with open(
            f"{self.datapath}/tinyshakespeare.txt",
            'r',
            encoding='utf-8') as f:

            text = f.read()
        
        self.data = text