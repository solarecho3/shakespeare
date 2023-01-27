import os
import logging
import yaml
from typing import Any

class DataLoader:

    def __init__(self, datapath: str | os.PathLike="../data/"):
        """
        A class to load data. References a configuration file, ../data/datasets.yml, to lookup data set paths.

        Parameters
        ----------

        path: str | os.PathLike - Relative path to the data folder. Default = ../data/
        """
        
        self.configpath = os.path.abspath('../config/')
        
        with open(f'{self.configpath}/datasets.yml', 'r', encoding='utf-8') as f:

            self.configs = yaml.load(f, Loader=yaml.Loader)

    def load(self, **kw):
        """
        Load a data set.

        Parameters
        ----------
        data: str
            'tiny', 'complete', default=None

        Example
        -------

        dataloader.load(data='tiny')
        """

        kw = kw.pop('data', None)

        if kw is None:

            print('Specify a data set.')

        elif kw is not None:

            with open(self.configs[kw]['path'], 'r', encoding='utf-8') as f:

                self.validate_schema(self.configs[kw], 'text')
                
                self.data = f.read()


    def validate_schema(self, f, schema):
        """
        Validate a data set schema.

        Parameters
        ----------

        f: the self.configs attribute with subscriptable keyword specifying the data set

        Example
        -------

        self.validate_schema(self.configs[kw], 'text')
        # self.configs[kw] = the data set schema specified in the configuration file
        # text = the desired schema for which to validate
        """
        
        if schema == f['schema']:

            # validate the schema here
            
            print('Text schema validated.')

        elif schema != f['schema']:

            print('Schema mismatch. Check arguments or configuration.')

        else:

            print('Invalid schema.')
