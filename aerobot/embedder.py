from Bio import SeqIO
from aerobot.utils import get_aa_kmers, get_nt_kmers
import os 
from typing import List, NoReturn
import pandas as pd 
import numpy as np 

class KmerEmbedder():

    @staticmethod
    def parse_fasta(path:str):
        ids, seqs = [], []
        for record in SeqIO.parse(path, 'fasta'):
            seqs.append(str(record.seq))
            # ids.append(record.id)
        return seqs

    def __init__(self, k:int, seq_type:str='nt'):
        self.k = k
        self.kmers = get_aa_kmers(k) if (seq_type == 'aa') else get_nt_kmers(k)
        self.counts = []
        self.ids = []

    def add_genome(self, path:str) -> NoReturn:
        file_name = os.path.basename(path) # Assuming these are the genome IDs. 

        counts = {kmer:0 for kmer in self.kmers} # Initialize the dictionary of k-mer counts. 
        seqs = KmerEmbedder.parse_fasta(path)
        for seq in seqs:
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i: i + self.k] # Extract the k-mer from the sequence. 
                if (len(kmer) == self.k) and (kmer in self.kmers):
                    counts[kmer] += 1 # Increment the k-mer's count.

        self.counts.append(counts)
        self.ids.append(file_name)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.counts, index=self.ids)
        df.index.name = 'genome_id'
        return df[self.kmers] # Make sure the order of columns is consistent. 

    def to_numpy(self) -> np.ndarray:
        df = self.to_dataframe(normalize=normalize)
        return df.values

    def to_csv(self, path:str):
        df = self.to_dataframe()
        df.to_csv(path)


