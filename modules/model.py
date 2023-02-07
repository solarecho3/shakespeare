'''
TODO Improve the logger
TODO Schema validator
'''
from __future__ import annotations

import os
import sys
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
            logging.exception(f'Exception raised in {func.__name__}:\n{str(e)}')
            raise SystemExit(1)
    return wrapper

class HyperParams:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_iterations = 17_500
    evaluation_interval = 1000
    block_size = 32
    batch_size = 8
    trng_pct = .90
    learning_rate = 1e-3
    manual_seed = 7561
    embedding_table_dims = 32

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
            raise KeyError('DataLoader constructor called with improper arguments.')

        elif kwargs is not None:
            try:
                with open(Data.data_configurations[kwargs]['path'], 'r', encoding='utf-8-sig') as data_file:
                    Data.data = data_file.read()
                    logging.info(f"Data loaded from {data_file}...")
            except KeyError:
                raise KeyError('DataLoader constructor called with improper arguments.')
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
            raise KeyError("VocabularyConfigurer constructor called with improper arguments.")
    
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
            raise AttributeError("Vocabulary failed to create. Check arguments for the VocabularyConfigurer constructor.")

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
        logging.info(f'Data decoded using {Data.vocabulary_type} tokenization: {Data.decoded_data[:500]}...')

class Trainer:
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
    def __init__(self):
        
        # encode the data
        self.encoded_data = torch.tensor(Data.encoded_data, dtype=torch.long)
        
        self.training_set_percentage = HyperParams.trng_pct
        self.training_validation_pivot_point = int(self.training_set_percentage*len(self.encoded_data))
        
        self.training_data = self.encoded_data[:self.training_validation_pivot_point]
        self.validation_data = self.encoded_data[self.training_validation_pivot_point:]

        # the context length
        self.block_size = HyperParams.block_size

        # number of parallel processes
        self.batch_size = HyperParams.batch_size

        self.vocab_size = Data.vocabulary_size

        self.manual_seed = torch.manual_seed(HyperParams.manual_seed)
        logging.info(f'{self.manual_seed.seed}: {HyperParams.manual_seed}')

    # @logger
    def get_batch(self, train=True):

        data = self.training_data if train == True else self.validation_data

        # generate random int for start points in data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        
        # setup the time dimension, x-axis, for inputs
        x = torch.stack([data[i:i+self.block_size] for i in ix])

        # setup the targets y for inputs x
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        # send x,y to cuda device
        x, y = x.to(HyperParams.device), y.to(HyperParams.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.m.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(HyperParams.evaluation_interval)
            for k in range(HyperParams.evaluation_interval):
                X, Y = self.get_batch(split)
                logits, loss = self.m(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.m.train()
        return out

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

        xb, yb = self.get_batch()

        for b in range(self.batch_size): # batch or y-dimension
            for t in range(self.block_size): # time or x-dimension
                context = xb[b, :t+1]
                target = yb[b,t]

        # create language model
        # TODO replace this with a decoupled option for multiple language models
        _model = BigramLanguageModel()
        # send model to 'cuda' device
        self.m = _model.to(HyperParams.device)
        
        logits, loss = self.m(xb, yb)
        logging.info(f'Idealized loss: {numpy.log(self.vocab_size)}')
        logging.info(f'Loss: {loss}')

        # start optimization
        # TODO decouple this, move to function or module
        optimizer = torch.optim.AdamW(self.m.parameters(), lr=1e-3)

        for iteration in range(HyperParams.max_iterations):

            if iteration %  HyperParams.evaluation_interval == 0:
                losses = self.estimate_loss()
                logging.info(f'Step {iteration}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')
            
            # sample a batch of data
            xb, yb = self.get_batch(train=True)
            
            # evaluate the loss
            logits, loss = self.m(xb, yb)

            # zero the gradients
            optimizer.zero_grad(set_to_none=True)
            
            # get gradients of all parameters
            loss.backward()

            # use gradients to update new parameters
            optimizer.step()
        
        print(loss.item())

        context = torch.zeros((1,1), dtype=torch.long, device=HyperParams.device)

        # DataTrainer.generation
        Decoder(self.m.generate(context, max_new_tokens=500)[0].tolist())
        Data.generation = Data.decoded_data

class Head(torch.nn.Module):
    """
    The self-attention head.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(HyperParams.embedding_table_dims, head_size, bias=False)
        self.query = torch.nn.Linear(HyperParams.embedding_table_dims, head_size, bias=False)
        self.value = torch.nn.Linear(HyperParams.embedding_table_dims, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(HyperParams.block_size, HyperParams.block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.nn.functional.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(torch.nn.Module):
    """
    The heads network.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(HyperParams.embedding_table_dims, HyperParams.embedding_table_dims),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

# TODO https://youtu.be/kCc8FmEb1nY?t=5210

class BigramLanguageModel(torch.nn.Module):
    
    def __init__(self):

        logging.info('Initializing BigramLanguageModel...')
        super().__init__()

        # creates tensor of shape vocab_size x vocab_size
        self.token_embedding_table = torch.nn.Embedding(Data.vocabulary_size, HyperParams.embedding_table_dims)
        logging.info(f'Embedding table created: {self.token_embedding_table} with vocabulary size {Data.vocabulary_size}')

        self.position_embedding_table = torch.nn.Embedding(HyperParams.block_size, HyperParams.embedding_table_dims)
        self.self_attention_heads = MultiHeadAttention(4, HyperParams.embedding_table_dims//4) # 4 heads of 8-dimensional self-attention
        self.feed_forward = FeedForward()
        self.language_modeling_head = torch.nn.Linear(HyperParams.embedding_table_dims, Data.vocabulary_size)

    # @logger
    def forward(self, idx, targets=None):
        '''
        Logits provide the context by allowing each token
        to predict the next likely token, wrapping the results
        in a tensor of (B,T,C) shape
        '''

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integer
        # B = Batch or y-dimension, T = Time or x-dimension
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=HyperParams.device)) # (T,C)
        x = token_embeddings + positional_embeddings
        x = self.self_attention_heads(x)
        x = self.feed_forward(x)
        logits = self.language_modeling_head(x) # (B,T,vocabulary size)

        if targets is None:
            loss = None

        else:
            # reshape the logits for torch cross_entropy functional
            
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # 2D tensor, with B*T in 1D, C in 1D
            targets = targets.view(B*T) # 1D tensor, with B*T
            
            # interpret the distance from the target for the logit
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop idx to the last HyperParams.block_size tokens
            idx_cond = idx[:, -HyperParams.block_size:]

            # get predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

def clean_logs():
    pass

if __name__ == "__main__":
    DataLoader(data='')
    VocabularyConfigurer("char")
    Encoder(Data.data)
    Trainer().train()
    clean_logs()
