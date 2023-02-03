'''
TODO Improve the logger
TODO Schema validator
'''
from __future__ import annotations

import os
import yaml
import numpy
import torch
import logging
import functools

from typing import Any
from dataclasses import dataclass

logging.basicConfig(
    filename='../logs/model.log',
    encoding='utf-8',
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.DEBUG
    )
logging.warning(f"======= {__name__} START =======")

def logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        try:
            # prevent entire dataset from being printed to logs by slicing str(args)
            logging.info(f'{func} called with args: {str(args)[:150]}, kwargs: {kw}...')
            return func(*args, **kw)
        except Exception as e:
            logging.exception(f'Exception raised in {func.__name__}. Exception: {str(e)}')
    return wrapper

class HyperParams:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_iterations = 13_000
    evaluation_interval = 1000
    block_size = 32
    batch_size = 8
    manual_seed = 7561
    trng_pct = .90
    learning_rate = 1e-3

@dataclass
class Data:
    """
    All classes hand transmissible objects to this class.
    This prevents making one thread of all classes the model.
    """

    data: DataLoader.dataset
    data_configurations: DataLoader.__init__
    
    vocabulary: VocabularyConfigurer.create_vocabulary
    vocabulary_type: VocabularyConfigurer.__init__
    vocabulary_size: VocabularyConfigurer.create_vocabulary
    
    encoded_data: Encoder.__init__
    decoded_data: Decoder.__init__
    
    trng_pivot: Trainer.__init__
    training_data: Trainer.__init__
    validation_data: Trainer.__init__

    model: BigramLanguageModel.__init__
    optimizer: Trainer.train
    generation: Trainer.train
    token_embedding_table: BigramLanguageModel.__init__

class DataLoader:

    @logger
    def __init__(self, path: os.PathLike="../config/", **kwargs) -> DataLoader:
        """
        Initialize the DataLoader, load the datasets configuration file.
        """

        path = os.path.join(os.path.abspath(path), 'datasets.yml')

        with open(os.path.abspath(path), 'r', encoding='utf-8') as config_file:
            Data.data_configurations = yaml.load(config_file, Loader=yaml.Loader)

        kwargs = kwargs.pop('data', None)

        if kwargs is None:
            print(f'Specify a data set. Refer to {path} to configure data sets for loading.')

        elif kwargs is not None:
            with open(Data.data_configurations[kwargs]['path'], 'r', encoding='utf-8-sig') as data_file:
                Data.data = data_file.read()
                logging.info(f"Data loaded from {data_file}...")

    @logger
    def validate_schema(self):
        ...

class VocabularyConfigurer:

    @logger
    def __init__(self, type):
        """
        Configure the vocabulary.

        Parameters:
            type[str]: "char"
        """
        
        if type == "char":
            Data.vocabulary_type = "char"
        else:
            logging.critical("Provide a vocabulary type: [\"char\"]")
    
        self._create_vocabulary()

    @logger
    def _create_vocabulary(self):
        """
        Create the vocabulary for tokenization.
        """
        
        if Data.vocabulary_type == "char":
            Data.vocabulary = sorted(list(set(Data.data)))
            Data.vocabulary_size = len(Data.vocabulary)
            logging.info(f"Vocabulary created: {Data.vocabulary[:50]}")
        
        else:
            logging.error("Vocabulary not created.")

class Encoder:

    @logger
    def __init__(self, text):
        """
        Encode a text, using the configured vocabulary.

        Parameters:
            text[str]: Data.data
        """

        if Data.vocabulary_type == "char":
            stoi = {ch:i for i,ch in enumerate(Data.vocabulary)}
            encode = lambda s: [stoi[c] for c in s]

        Data.encoded_data = encode(text)
        logging.info(f'Data encoded using {Data.vocabulary_type} tokenization: {Data.encoded_data[:50]}...')

class Decoder:

    @logger
    def __init__(self, text):
        """
        Decode a text, using the configured vocabulary.

        Parameters:
            text[str]: Data.encoded_data
        """

        if Data.vocabulary_type == "char":
            itos = {i:ch for i,ch in enumerate(Data.vocabulary)}
            decode = lambda l: "".join([itos[i] for i in l])

        Data.decoded_data = decode(text)
        logging.info(f'Data decoded using {Data.vocabulary_type} tokenization: {Data.decoded_data[:50]}...')

class Trainer:

    @logger
    def __init__(self):

        Data.encoded_data = torch.tensor(Data.encoded_data, dtype=torch.long)
        
        # calculate the pivot point for training vs. validation data
        Data.trng_pivot = int(HyperParams.trng_pct*len(Data.encoded_data))
        logging.info(f'Training set percentage set to: {HyperParams.trng_pct}({HyperParams.trng_pct*100}%)...')

        Data.training_data = Data.encoded_data[:Data.trng_pivot]
        Data.validation_data = Data.encoded_data[Data.trng_pivot:]

    @logger
    def get_batch(self, train=True):
        
        data = Data.training_data if train == True else Data.validation_data

        # generate random int for start points in data
        ix = torch.randint(len(data) - HyperParams.block_size, (HyperParams.batch_size,))

        # the time dimension, x-axis, for inputs
        x = torch.stack([data[i:i+HyperParams.block_size] for i in ix])

        # the targets y for inputs x
        y = torch.stack([data[i+1:i+HyperParams.block_size+1] for i in ix])

        # send to GPU
        x, y = x.to(HyperParams.device), y.to(HyperParams.device)

        return x, y

    @logger
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        Data.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(HyperParams.evaluation_interval)
            for k in range(HyperParams.evaluation_interval):
                X, Y = self.get_batch(split)
                logits, loss = Data.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        Data.model.train()
        return out

    @logger
    def train(self):

        x = Data.training_data[:HyperParams.block_size]
        y = Data.training_data[1:HyperParams.block_size+1]

        for t in range(HyperParams.block_size):
            context = x[:t+1]
            target = y[t]

        xb, yb = self.get_batch()

        for b in range(HyperParams.batch_size):
            for t in range(HyperParams.block_size):
                context = xb[b, :t+1]
                target = yb[b,t]

        model = BigramLanguageModel()

        model = model.to(HyperParams.device)

        logits, loss = model(xb, yb)
        logging.info(f'Idealized loss: {numpy.log(Data.vocabulary_size)}')
        logging.info(f'Loss: {loss}')

        Data.optimizer = torch.optim.AdamW(Data.model.parameters(), lr=HyperParams.learning_rate)

        for iteration in range(HyperParams.max_iterations):
            if iteration % HyperParams.evaluation_interval == 0:
                losses = self.estimate_loss()
                logging.info(f'Step {iteration}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')
            
            # sample a batch of data
            xb, yb = self.get_batch()

            # evaluate the loss
            logits, loss = model(xb, yb)

            # zero the grads
            Data.optimizer.zero_grad(set_to_none=True)

            # get gradients of all parameters
            loss.backward()

            # use gradients to update new parameters
            Data.optimizer.step()
        
        logging.info(f'Loss item: {loss.item()}')

        context = torch.zeros((1,1), dtype=torch.long, device=HyperParams.device)

        Decoder(model.generate(context, max_new_tokens=500)[0].tolist())
        logging.warning(f'Generation: {Data.decoded_data}')

class BigramLanguageModel(torch.nn.Module):

    @logger
    def __init__(self):
        super().__init__()

        Data.token_embedding_table = torch.nn.Embedding(Data.vocabulary_size, Data.vocabulary_size)
        logging.info(f'Embedding table created: {Data.token_embedding_table} with vocabulary size {Data.vocabulary_size}...')
    
    @logger
    def forward(self, idx, targets=None):

        logits = Data.token_embedding_table(idx)

        if targets is None:
            loss = None

        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = torch.nn.functional.cross_entropy(logits, targets)
        
        return logits, loss
    
    @logger    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self(idx)

            logits = logits[:, -1, :]

            probs = torch.nn.functional.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx