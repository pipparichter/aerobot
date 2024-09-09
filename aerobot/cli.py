import os
from aerobot.utils import DATA_DIR, MODELS_DIR, FEATURE_TYPES
from aerobot.classifiers import TernaryClassifier, BinaryClassifier, Classifier
from aerobot.embedder import KmerEmbedder
from aerobot.dataset import FeatureDataset
import datetime
import wget 
import argparse
import subprocess
import json
import pickle
from tqdm import tqdm
import numpy as np 
import pandas as pd 
# import warnings 
# import glob

# warnings.simplefilter('ignore')


def embed():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default=None, type=str)
    parser.add_argument('--output-path', '-o', default=None, type=str)
    parser.add_argument('--feature-type', default=None, choices=FEATURE_TYPES, type=str)
    args = parser.parse_args()

    seq_type, k = args.feature_type.split('_')
    k = int(k[0])
    embedder = KmerEmbedder(k, seq_type=seq_type)
   
    if os.path.isdir(args.input_path):
        file_names = os.listdir(args.input_path)
        paths = [os.path.join(args.input_path, file_name) for file_name in file_names] 
        for path in tqdm(paths, desc='embed: Embedding genome files...'):
            embedder.add_genome(path)
    elif os.path.isfile(args.input_path):
        embedder.add_genome(args.input_path)

    embedder.to_csv(args.output_path) # Save the embeddings to the output path. 
    print(f'embed: {k}-mer embeddings saved to {args.output_path}.')


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default=None) 
    parser.add_argument('--input-path', default=None, type=str)
    parser.add_argument('--embedding-path', default=None, type=str)
    parser.add_argument('--output-path', default=None, type=str)
    parser.add_argument('--feature-type', default=None, choices=FEATURE_TYPES, type=str)
    parser.add_argument('--normalize', type=bool, default=True)
    # parser.add_argument('--seq-type', '-s', default='aa', choices=['nt', 'aa'], type=str)
    # parser.add_argument('--kmer-length', '-k', default=3, type=int)
    args = parser.parse_args()


    if (args.embedding_path is not None) and (args.input_path is None):
        print(f'predict: Loading {args.feature_type} embeddings from {args.embedding_path}.')
        dataset = FeatureDataset.from_csv(args.embedding_path, normalize=args.normalize, feature_type=args.feature_type)
    else: # If an embedding file is not specified, but an input file is, then go ahead and generate embeddings.
        output_dir = os.path.dirname(args.output_path)
        embedding_file_name = f'{args.feature_type}_embeddings.csv' 
        embedding_path = os.path.join(output_dir, embedding_file_name) if (args.embedding_path is None) else args.embedding_path
        subprocess.run(f'embed --input-path {args.input_path} --output-path {embedding_path} --feature-type {args.feature_type}', shell=True, check=True)
        dataset = FeatureDataset.from_csv(embedding_path, normalize=args.normalize)
    
    # TODO: Add a check to make sure that the dataset is normalized if the model was trained on normalized data. 
    model = Classifier.load(os.path.join(MODELS_DIR, args.model_name + '.pkl'))
    y_pred = model.predict(dataset.to_numpy()[0])
    y_pred_df = pd.DataFrame(y_pred, index=dataset.index)
    y_pred_df.index.name = 'genome_id'
    y_pred_df.to_csv(args.output_path)
    print(f'predict: Predictions saved to {args.output_path}.')


def models():
    '''Display all available models.'''
    model_names = os.listdir(MODELS_DIR)
    model_names = [model_name.replace('.pkl', '') for model_name in model_names if '.pkl' in model_name]
    print('models: Available models')
    for model_name in model_names:
        print(f'\t{model_name}')



def download():
    '''Download training, testing, and validation data from Figshare.'''
    train_path = os.path.join(DATA_DIR, 'training_datasets.h5')
    val_path = os.path.join(DATA_DIR, 'validation_datasets.h5')
    test_path = os.path.join(DATA_DIR, 'testing_datasets.h5')

    # Validation data https://figshare.com/ndownloader/files/48991618
    # Testing data https://figshare.com/ndownloader/files/48991621
    # Training data https://figshare.com/ndownloader/files/48991624

    for file_name in os.listdir(DATA_DIR): # Remove existing files, or they keep getting re-downloaded.
        if (file_name != '__init__.py') and os.path.isfile(os.path.join(DATA_DIR, file_name)):
            os.remove(os.path.join(DATA_DIR, file_name))

    wget.download('https://figshare.com/ndownloader/files/48991618', val_path)
    print(f'\ndownload: Downloaded validation datasets to {val_path}.')
    wget.download('https://figshare.com/ndownloader/files/48991621', test_path)
    print(f'\ndownload: Dowloaded testing datasets to {test_path}.')
    wget.download('https://figshare.com/ndownloader/files/48991624', train_path)
    print(f'\ndownload: Dowloaded training datasets to {train_path}.')


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', type=str, default='nt_5mer')
    parser.add_argument('--results-dir', default=os.path.expanduser('~'), type=str)
    # parser.add_argument('--data-dir', default=DATA_DIR, type=str)
    # parser.add_argument('--models-dir', default=MODELS_DIR, type=str)
    parser.add_argument('--binary', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--normalize', type=bool, default=True)
    args = parser.parse_args()

    train_path = os.path.join(DATA_DIR, 'training_datasets.h5')
    val_path = os.path.join(DATA_DIR, 'validation_datasets.h5')
    test_path = os.path.join(DATA_DIR, 'testing_datasets.h5')

    if (not os.path.exists(train_path)) or (not os.path.exists(val_path)) or (not os.path.exists(test_path)):
        subprocess.run(f'download --data-dir {DATA_DIR}', shell=True, check=True)

    train_dataset = FeatureDataset.from_hdf(train_path, feature_type=args.feature_type, normalize=args.normalize)
    val_dataset = FeatureDataset.from_hdf(val_path, feature_type=args.feature_type, normalize=args.normalize)
    test_dataset = FeatureDataset.from_hdf(test_path, feature_type=args.feature_type, normalize=args.normalize)

    model = BinaryClassifier(input_dim=train_dataset.dim) if args.binary else TernaryClassifier(input_dim=train_dataset.dim)
    train_losses, val_accs, best_epoch = model.fit(*train_dataset.to_numpy(), *val_dataset.to_numpy(), batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)
    test_acc = model.accuracy(*test_dataset.to_numpy())
    print(f'train: {np.round(test_acc * 100, 2)}% accuracy on testing dataset.')

    train_results = dict()
    train_results['train_losses'] = train_losses
    train_results['val_accs'] = val_accs
    train_results['batch_size'] = args.batch_size
    train_results['epochs'] = args.epochs
    train_results['lr'] = args.lr
    train_results['best_epoch'] = best_epoch
    train_results['binary'] = args.binary
    train_results['test_acc'] = test_acc

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    train_results_path = os.path.join(args.results_dir, f'train_{time_stamp}.json')
    # train_plot_path = os.path.join(args.results_dir, f'train_{time_stamp}.png')
    # Save the results of the training run. 
    with open(train_results_path, 'w') as f:
        json.dump(train_results, f)
    print(f'train: Saved training results to {train_results_path}')

    # Also save the trained model. 
    model_name = f"{'binary' if args.binary else 'ternary'}_model_{args.feature_type}_{time_stamp}.pkl"
    model_path = os.path.join(MODELS_DIR, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'train: Saved trained model to {model_path}')


def config():
    pass 


