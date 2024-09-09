import os 
from typing import List
import pandas as pd 
import numpy as np 
import json 
import pickle
from aerobot import data
from aerobot import models
import itertools
import torch
import io
import pandas as pd 

MODELS_DIR = os.path.dirname(os.path.abspath(models.__file__))
DATA_DIR = os.path.dirname(os.path.abspath(data.__file__))

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'U']
NUCLEOTIDES = ['A', 'C', 'T', 'G']

FEATURE_TYPES = [f'nt_{i}mer' for i in range(1, 6)] + [f'aa_{i}mer' for i in range(1, 4)]

def get_aa_kmers(k:int):
    aa_kmers = [''.join(i) for i in itertools.product(AMINO_ACIDS, repeat=k)]
    return sorted(aa_kmers)

def get_nt_kmers(k:int):
    nt_kmers = [''.join(i) for i in itertools.product(NUCLEOTIDES, repeat=k)]
    return sorted(nt_kmers)

class Unpickler(pickle.Unpickler):
    '''For un-pickling models which were trained on GPUs. See https://github.com/pytorch/pytorch/issues/16797'''
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), weights_only=False, map_location='cpu')
        else: return super().find_class(module, name)


class NumpyEncoder(json.JSONEncoder):
    '''Encoder for converting numpy data types into types which are JSON-serializable. Based
    on the tutorial here: https://medium.com/@ayush-thakur02/understanding-custom-encoders-and-decoders-in-pythons-json-module-1490d3d23cf7'''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)



def plot_training_curve(training_results_path:str=None, output_path:str=None):

    # Fields in training data results are epoch, train_losses, val_accs, and best_epoch. 
    pass 