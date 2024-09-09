from typing import List
from aerobot.utils import get_aa_kmers, get_nt_kmers
import itertools
import re
import pandas as pd 
import numpy as np 


class FeatureDataset():

    def __init__(self, features:pd.DataFrame, labels:pd.DataFrame=None, feature_type:str=None, normalize:bool=True):

        self.feature_type = feature_type
        self.aa_or_nt, k = feature_type.split('_')
        self.k = int(k[0])

        if labels is not None:
            features, self.labels = features.align(labels, join='left', axis=0)
        else:
            self.labels = None

        features = features.fillna(0) # Get rid of any NaNs in the data.
        kmers = get_aa_kmers(self.k) if (self.aa_or_nt == 'aa') else get_nt_kmers(self.k)
        features = features.drop(columns=[col for col in features.columns if col not in kmers])
        missing_features = pd.DataFrame(0, index=features.index, columns=[kmer for kmer in kmers if (kmer not in features.columns)])
        features = pd.concat([features, missing_features], axis=1)
        self.index = features.index
        self.features = features[kmers] # Make sure the ordering is consistent.
        self.dim = len(kmers) # Set the number of features. 

        if normalize:
            self.features = self.features.apply(lambda row : row / row.sum(), axis=1)

    @classmethod
    def from_csv(cls, path:str, feature_type:str=None, index_col:str='genome_id', normalize:bool=True):
        features = pd.read_csv(path).set_index(index_col)
        return FeatureDataset(features, labels=None, feature_type=feature_type)

    @classmethod
    def from_hdf(cls, path:str, feature_type:str=None, normalize:bool=True):
        with pd.HDFStore(path, 'r') as store:
            features = store.get(feature_type)
            labels = store.get('labels') if '/labels' in store.keys() else None
        return FeatureDataset(features, labels=labels, feature_type=feature_type)

    def to_numpy(self, binary:bool=False):
        '''Convert the underlying dataset to Numpy arrays. Return labels if the FeatureDataset is labeled.'''
        if self.labels is not None:
            labels = self.labels['binary'].values if binary else self.labels['ternary'].values
        else:
            labels = None
        return self.features.values, labels




