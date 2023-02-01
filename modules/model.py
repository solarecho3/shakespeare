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
TODO Improve the logger
'''

import os
import logging
import yaml
from typing import Any
import torch
import tiktoken
import functools

logging.basicConfig(
    filename='../logs/model.log',
    encoding='utf-8',
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.DEBUG
    )
logging.info(f"======= {__name__} START =======")

def logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        logging.info(f'{func} called with args: {str(args)[:150]}, kwargs: {kw}...')
        logging.info('\n')
        return func(*args, **kw)
    return wrapper

class DataEventQueue:
    """
    Multi-threading support for a data event processing queue.
    
    time, priority, action, argument, kwargs
    """

    @logger
    def __init__(self):
        self.queue = []

    @logger
    def append_event(self, event):
        """Add an event to the queue."""
        pass
    
    @logger
    def pop_event(self, queue):
        """Pop from left."""
        pass

class DataLoader:
    """
    A class to load data. References a configuration file, ../data/datasets.yml, to lookup data set paths.

    Parameters
    ----------

    path: str | os.PathLike - Relative path to the data folder. Default = ../data/
    """

    @logger
    def __init__(self, datapath: str | os.PathLike="../data/"):

        self.configpath = os.path.abspath('../config/')
        
        with open(f'{self.configpath}/datasets.yml', 'r', encoding='utf-8') as f:

            self.configs = yaml.load(f, Loader=yaml.Loader)

    @logger
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

    @logger
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
    
    @logger
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
            self.vocabulary = self.create_subword_vocab(data)
        elif vocab == "word":
            self.vocabulary = self.create_word_vocab(data, **kw)
        else:
            print("Provide a vocabulary type: ", ["char", "subword", "word"])

    @logger
    def create_subword_vocab(self, data):
        """
        Parse and tokenize a subword-based vocabulary using Tiktoken BPE.

        EST (complete works) = 50527
        """
        subword = tiktoken.get_encoding("gpt2")
        self.vocab_type = "subword"

        return subword

    @logger
    def create_character_vocab(self, data):
        """
        Parse and tokenize a character-based vocabulary
        from a DataLoader object.
        """

        char = sorted(list(set(data.data)))
        self.vocab_type = "char"

        return char

    @logger
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

        word = sorted(list(set(data.data.split(maxsplit=-1))))
        self.vocab_type = "word"

        return word

    @logger
    def encode_vocabulary(self, text):
        """
        Encode a vocabulary with simple enumeration.
        """

        if self.vocab_type == "subword":
            encode = tiktoken.get_encoding("gpt2").encode(text)
            return encode

        stoi = {ch:i for i,ch in enumerate(self.vocabulary)}
        
        if self.vocab_type == "word":
            encode = lambda s: [stoi[c] for c in s.split(maxsplit=-1)]
        elif self.vocab_type == "char":
            encode = lambda s: [stoi[c] for c in s]

        return encode(text)

    @logger
    def decode_vocabulary(self, text, spaces=False):
        """
        Decode a vocabulary with simple enumeration.

        Parameters
        ----------
        spaces: bool - Whether or not to insert whitespace between words.
        """

        if self.vocab_type == "subword":
            decode = tiktoken.get_encoding("gpt2").decode(text)
            return decode

        itos = {i:ch for i,ch in enumerate(self.vocabulary)}
        
        if spaces:
            decode = lambda l: " ".join([itos[i] for i in l])
        else:
            decode = lambda l: "".join([itos[i] for i in l])

        return decode(text)

class DataTrainer:
    """
    Prepare and executing training.

    Parameters
    ----------
    vocab: DataParser - the DataParser object with vocabulary created.
    data: DataLoader - the DataLoader object with data loaded.
    training_set_percentage: float - Percentage to dedicate to training, in decimal[float] form.
    block_size: int - Size of each training block.
    """

    @logger
    def __init__(self, vocab: DataParser, data: DataLoader, training_set_percentage: float, block_size: int=8, batch_size: int=4):
        
        # encode the data
        self.encoded_data = torch.tensor(vocab.encode_vocabulary(data.data), dtype=torch.long)
        
        self.training_set_percentage = training_set_percentage
        self.training_validation_pivot_point = int(training_set_percentage*len(self.encoded_data))
        
        self.training_data = self.encoded_data[:self.training_validation_pivot_point]
        self.validation_data = self.encoded_data[self.training_validation_pivot_point:]

        # the context length
        self.block_size = block_size

        # number of parallel processes
        self.batch_size = batch_size
        
        self.manual_seed = torch.manual_seed(7561)
        logging.info(f'torch.manual_seed({self.manual_seed.seed})')

    @logger
    def get_batch(self, train=True):

        data = self.training_data if train == True else self.validation_data

        # generate random int for start points in data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        
        # setup the time dimension, x-axis, for inputs
        x = torch.stack([data[i:i+self.block_size] for i in ix])

        # setup the targets y for inputs x
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y

    @logger
    def train(self):
        """
        "Time dimension", as in, input of length n or O(n)
        
        Chunk a block of 9 characters,
        x: inputs
        y: targets

        context: x to and including tth char
        target: y at the tth char
        """

        x = self.training_data[:self.block_size]
        y = self.training_data[1:self.block_size+1]
        
        for t in range(self.block_size):
            context = x[:t+1]
            target = y[t]
            print(f'when input is {context} the target is: {target}')

        xb, yb = self.get_batch()
        logging.info(f'inputs:')
        logging.info(xb.shape)
        logging.info(xb)
        logging.info(f'targets:')
        logging.info(yb.shape)
        logging.info(yb)

        for b in range(self.batch_size): # batch or y-dimension
            for t in range(self.block_size): # time or x-dimension
                context = xb[b, :t+1]
                target = yb[b,t]
                logging.info(f'Input: {context.tolist()}\nTarget: {target}')
