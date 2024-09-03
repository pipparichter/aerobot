from typing import List
from aerobot.utils import AMINO_ACIDS, NUCLEOTIDES
import itertools
import re

def get_aa_kmers(k:int):
    aa_kmers = [''.join(i) for i in itertools.product(AMINO_ACIDS, repeat=k)]
    return sorted(aa_kmers)

def get_nt_kmers(k:int):
    nt_kmers = [''.join(i) for i in itertools.product(NUCLEOTIDES, repeat=k)]
    return sorted(nt_kmers)


class FeatureDataset():

    def __init__(self, features:pd.DataFrame, labels:pd.DataFrame=None, feature_type:str=None, normalize:bool=True):

        self.feature_type = feature_type
        self.aa_or_nt, k = feature_type.split('_')
        self.k = int(k[0])

        if labels is not None:
            features, self.labels = features.align(self.labels, join='left', axis=0)
        else:
            self.labels = None

        features = features.fillna(0) # Get rid of any NaNs in the data.
        kmers = get_aa_kmers(self.k) if (self.aa_or_nt == 'aa') else get_nt_kmers(self.k)
        features = features.drop(columns=[col for col in features.columns if col not in kmers])
        missing_features = pd.DataFrame(0, index=features.index, columns=[kmer for kmer in kmers if (kmer not in features.columns)])
        features = pd.concat([features, missing_features], axis=1)
        self.features = features[[kmers]] # Make sure the ordering is consistent.
        self.dim = len(kmers) # Set the number of features. 

        if normalize:
            self.features = self.features.apply(lambda row : row / row.sum(), axis=1)

    def from_csv(self, path:str, feature_type:str=None, index_col:str='genome_id'):
        features = pd.read_csv(path).set_index(index_col)
        return FeatureDataset(features, labels=None, feature_type=feature_type)

    def from_hdf(self, path:str, feature_type:str=None):
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
        return features.values, labels




