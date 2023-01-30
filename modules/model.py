'''
# Shakespeare GPT model
This is the model. It is designed to run asynchronously
from the presenter, in order to continually update or
create new models for presentation.

# MV* format
Each atomic data process will be performed by a different
class:

| Load       | Parse      | Train       | Configure      | Provide      | Log and Time |
|------------|------------|-------------|----------------|--------------|--------------|
| DataLoader | DataParser | DataTrainer | DataConfigurer | DataProvider | DataLogger   |

The model only retrieves data and performs actions upon it.
To provide data to a presenter, the presenter must request
the data from the model using the DataProvider class.

The data will be sent by the provider to the requestor in
a standardized data-interchange format such as JSON, XML, etc...

# Docstring format
This package follows the  numpy/scipy docstring format.

# Concurrence

The model module is built to concurrently and asynchronously handle
data. To do so, the model adds desired method calls to a priority queue,
intended to prevent race conditions. The priority queue processes data
tasks concurrently; GPU multi-processing for training and CPU
multi-processing and threading for all other tasks.
TODO Review this for exactness
'''

import os
import logging
import yaml
from typing import Any

logging.basicConfig(level=logging.INFO)

class DataEventQueue:
    """
    Multi-threading support for a data event processing queue.
    
    time, priority, action, argument, kwargs
    """

    def __init__(self):
        self.queue = []

    def append_event(self, event):
        """Add an event to the queue."""
    
    def pop_event(self, queue):
        """Pop from left."""

class DataLoader:
    """
    A class to load data. References a configuration file, ../data/datasets.yml, to lookup data set paths.

    Parameters
    ----------

    path: str | os.PathLike - Relative path to the data folder. Default = ../data/
    """

    def __init__(self, datapath: str | os.PathLike="../data/"):

        self.configpath = os.path.abspath('../config/')
        
        with open(f'{self.configpath}/datasets.yml', 'r', encoding='utf-8') as f:

            self.configs = yaml.load(f, Loader=yaml.Loader)

    def load(self, **kw):
        """
        Load a data set. Pass data=<dataset key> from the datasets.yml configuration file.

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

            print(f'Specify a data set. Refer to {self.configpath}/datasets.yml...')

        elif kw is not None:

            # the datasets config file is YAML
            # each data set is specified with:
            # <data set name>: { path: '', schema: '' } format
            with open(self.configs[kw]['path'], 'r', encoding='utf-8-sig') as f:

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

class DataParser:
    """
    A class to parse loaded data.
    """

    def __init__(self, vocab, data: DataLoader, **kw):
        """
        Parse the data from a DataLoader object.

        Must be bound to the DataLoader class
        in order to inherit the dataset.

        Parameters
        ----------
        vocab: str - The type of tokenization.
            "char" | "subword" | "word"

        data: DataLoader - The instantiated DataLoader class, with loaded data.
            
        **kw: kwargs - Pass kwargs to the tokenizer.
            strip=False - Strips special characters from word tokens.
        """

        if vocab == "char":
            self.vocabulary = self.create_character_vocab(data)
        elif vocab == "subword":
            ...
        elif vocab == "word":
            self.vocabulary = self.create_word_vocab(data, **kw)
        else:
            print("Provide a vocabulary type: ", ["char", "subword", "word"])

    def create_character_vocab(self, data):
        """
        Parse and tokenize a character-based vocabulary
        from a DataLoader object.

        This 
        """

        chars = sorted(list(set(data.data)))
        self.vocab_type = "chars"

        return chars

    def create_word_vocab(self, data, **kw):
        """
        Parse and tokenize a word-based vocabulary
        from a DataLoader object. Strips special characters
        if the model.DataParser() constructor contains
        a 'strip=True' argument.
        """

        if kw.get("strip", False):

            if kw["strip"]:

                for c in ["?", "!", ".", ",", "'s", "-", "_", "[", "]", "(", ")"]:
                    data.data = data.data.replace(c, "")

        words = sorted(list(set(data.data.split(maxsplit=-1))))
        self.vocab_type = "words"

        return words

    def encode_vocabulary(self, text):
        """
        Encode a vocabulary with simple enumeration.
        """

        stoi = {ch:i for i,ch in enumerate(self.vocabulary)}
        
        if self.vocab_type == "words":
            encode = lambda s: [stoi[c] for c in s.split(maxsplit=-1)]
        elif self.vocab_type == "chars":
            encode = lambda s: [stoi[c] for c in s]

        return encode(text)

    def decode_vocabulary(self, text, spaces=False):
        """
        Decode a vocabulary with simple enumeration.

        Parameters
        ----------
        spaces: bool - Whether or not to insert whitespace between words.
        """

        itos = {i:ch for i,ch in enumerate(self.vocabulary)}
        
        if spaces:
            decode = lambda l: " ".join([itos[i] for i in l])
        else:
            decode = lambda l: "".join([itos[i] for i in l])

        return decode(text)