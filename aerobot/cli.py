import os
from aerobot.utils import DATA_DIR, RESULTS_DIR, MODELS_DIR
from aerobot.models import TernaryClassifier, BinaryClassifier
from aerobot.dataset import FeatureDataset
import datetime
import wget 
import argparse
import subprocess
import json
import pickle

def predict():
    pass


def download():
    '''Download training, testing, and validation data from Figshare.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=DATA_DIR, type=str)
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, 'training_datasets.csv')
    val_path = os.path.join(args.data_dir, 'validation_datasets.csv')
    test_path = os.path.join(args.data_dir, 'testing_datasets.csv')

    # Validation data https://figshare.com/ndownloader/files/48921232
    # Testing data https://figshare.com/ndownloader/files/48921235
    # Training data https://figshare.com/ndownloader/files/48921238

    wget.download('https://figshare.com/ndownloader/files/48921232', val_path)
    print(f'download: Dowloaded validation datasets to {val_path}.')
    wget.download('https://figshare.com/ndownloader/files/48921235', test_path)
    print(f'download: Dowloaded testing datasets to {test_path}.')
    wget.download('https://figshare.com/ndownloader/files/48921238', train_path)
    print(f'download: Dowloaded training datasets to {train_path}.')


def train(results_path:str, data_dir:str=DATA_DIR, plot:bool=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', type=str, default='nt_5mer')
    parser.add_argument('--data-dir', default=DATA_DIR, type=str)
    parser.add_argument('--results-dir', default=RESULTS_DIR, type=str)
    parser.add_argument('--models-dir', default=MODELS_DIR, type=str)
    parser.add_argument('--binary', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, 'training_datasets.csv')
    val_path = os.path.join(args.data_dir, 'validation_datasets.csv')
    test_path = os.path.join(args.data_dir, 'testing_datasets.csv')

    if (not os.path.exists(train_path)) or (not os.path.exists(val_path)) or (not os.path.exists(test_path)):
        subprocess.run(f'download --data-dir {args.data_dir}', shell=True, check=True)


    train_dataset = FeatureDataset.from_hdf(train_path, feature_type=args.feature_type)
    val_dataset = FeatureDataset.from_hdf(val_path, feature_type=args.feature_type)

    model = BinaryClassifier(input_dim=train_dataset.dim) if args.binary else TernaryClassifier(input_dim=train_dataset.dim)
    train_losses, val_accs, best_epoch = model.fit(*train_dataset.to_numpy(), *val_dataset.to_numpy(), batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)

    train_results = dict()
    train_results['train_losses'] = train_losses
    train_results['val_accs'] = val_accs
    train_results['batch_size'] = args.batch_size
    train_results['epochs'] = args.epochs
    train_results['lr'] = args.lr
    train_results['best_epoch'] = best_epoch
    train_results['binary'] = args.binary

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    train_results_path = os.path.join(args.results_dir, f'train_{time_stamp}.json')
    # train_plot_path = os.path.join(args.results_dir, f'train_{time_stamp}.png')
    # Save the results of the training run. 
    with open(train_results_path, 'w') as f:
        json.dump(f, data)
    print(f'train: Saved training results to {train_results_path}')

    # Also save the trained model. 
    model_name = f"{'binary' if args.binary else 'ternary'}_model_{feature_type}_{time_stamp}.pkl"
    model_path = os.path.join(args.models_dir, model_name)
    with open(model_path, 'rb') as f:
        pickle.dump(model, f)
    print(f'train: Saved trained model to {model_path}')


def config():
    pass 


