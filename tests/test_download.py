import unittest 
from aerobot.utils import DATA_DIR, FEATURE_TYPES
from aerobot.dataset import FeatureDataset
from parameterized import parameterized
import os 
import pandas as pd 

TEST_SIZE = 587
TRAIN_SIZE = 2048
VAL_SIZE = 465

class TestDownload(unittest.TestCase):
    '''A series of tests to ensure the data download was successful.'''
    train_path = os.path.join(DATA_DIR, 'training_datasets.h5')
    val_path = os.path.join(DATA_DIR, 'validation_datasets.h5')
    test_path = os.path.join(DATA_DIR, 'testing_datasets.h5')

    paths = [train_path, val_path, test_path]
    sizes = {'training_datasets.h5':TRAIN_SIZE, 'testing_datasets.h5':TEST_SIZE, 'validation_datasets.h5':VAL_SIZE}

    @parameterized.expand(paths)
    def test_files_in_data_dir(self, path:str):
        '''Check to make sure all the dataset files were successfully installed in the data directory.'''
        self.assertTrue(os.path.exists(path))

    @parameterized.expand(paths)
    def test_datasets_are_labeled(self, path:str):
        '''Check to make sure that all feature types are present in the datasets.'''
        store = pd.HDFStore(path)
        keys = [key.replace('/', '') for key in store.keys()]
        store.close()
        file_name = os.path.basename(path)
        self.assertTrue('labels' in keys, msg=f'TestDownload.test_datasets_are_labeled: Missing labels in {file_name}.')

    @parameterized.expand(paths)
    def test_all_feature_types_present(self, path:str):
        '''Check to make sure that all feature types are present in the datasets.'''
        store = pd.HDFStore(path)
        keys = [key.replace('/', '') for key in store.keys()]
        store.close()
        file_name = os.path.basename(path)
        for feature_type in FEATURE_TYPES:
            self.assertTrue(feature_type in keys, msg=str(keys)) # msg=f'TestDownload.test_all_feature_types_present: Missing feature type {feature_type} in {file_name}.')

    @parameterized.expand(paths)
    def test_no_duplicate_entries(self, path:str):
        '''Check to make sure that there are no duplicate entries within each dataset.'''
        ids = pd.read_hdf(path, key='labels').index.values.tolist()
        self.assertTrue(len(ids) == len(set(ids)))

    @parameterized.expand(paths)
    def test_feature_type_sizes_are_equal(self, path:str):
        pass 
    
    @parameterized.expand(paths)
    def test_size_is_correct(self, path:str):
        pass 

    def test_datasets_are_disjoint(self):
        '''Check to make sure that the training, testing, and validation sets are disjoint (i.e. have no overlapping entries).'''
        ids = []
        for path in TestDownload.paths:
            ids += pd.read_hdf(path, key='labels').index.values.tolist()

        self.assertTrue(len(ids) == len(set(ids)))

    def test_feature_type_genome_ids_match(self):
        pass 


if __name__ == '__main__':
    unittest.main()







