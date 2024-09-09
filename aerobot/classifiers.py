
import torch 
import os 
import numpy as np 
import pickle
from sklearn.metrics import balanced_accuracy_score 
from aerobot.utils import Unpickler
from typing import Tuple
import sklearn
import copy
from tqdm import tqdm

class WeightedMSELoss(torch.nn.Module):

    def __init__(self):

        super(WeightedMSELoss, self).__init__()
        self.weights = 1
        # self.n_classes = n_classes
        self.categories = None

    def fit(self, y:np.ndarray, categories:np.ndarray=None):
        '''Compute the weights to use based on the frequencies of each class in the input Dataset.'''
        # Compute loss weights as the inverse frequency.
        self.weights = torch.FloatTensor([1 / (np.sum(y == c) / len(y)) for c in categories])

    def forward(self, outputs:torch.FloatTensor, targets:torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean((outputs - targets)**2 * self.weights)


class Classifier(torch.nn.Module):
    '''Two-layer neural network for binary or ternary classification classification.'''
    def __init__(self, input_dim:int=None, hidden_dim:int=512, output_dim:int=None):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__()
        # NOTE: SoftMax is applied automatically when using the torch CrossEntropy loss function. 
        # Because we are using a custom loss function, we need to apply SoftMax here.
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Softmax(dim=1)).to(self.device)

        self.loss_func = WeightedMSELoss()
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='error', sparse_output=False)
        self.n_classes = output_dim

    def forward(self, X:np.ndarray, one_hot:bool=True) -> torch.FloatTensor:
        # X = torch.FloatTensor(X).to(self.device) # Convert numpy array to Tensor. Make sure it is on the GPU, if available.
        outputs = self.classifier(X)
        if one_hot:
            outputs = torch.nn.functional.one_hot(torch.argmax(outputs, axis=1), num_classes=self.n_classes)
        return outputs

    @staticmethod
    def get_batches(X:torch.FloatTensor, y:torch.FloatTensor, batch_size:int=16) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''Create batches of size batch_size from training data and labels.'''
        # Don't bother with balanced batches. Doesn't help much with accuracy anyway.
        n_batches = len(X) // batch_size + 1
        return torch.tensor_split(X, n_batches, axis=0), torch.tensor_split(y, n_batches, axis=0)

    @staticmethod
    def shuffle(X:torch.FloatTensor, y:torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''Shuffle the input arrays without modifying them inplace.'''
        shuffle_idxs = np.arange(len(X)) # Get indices to shuffle the inputs and labels. 
        np.random.shuffle(shuffle_idxs)
        return X[shuffle_idxs, :], y[shuffle_idxs, :]

    def _fit(self, X:np.ndarray, y:np.ndarray):
        '''Helper function for fitting the standardizing scaler, one-hot encoder, and loss function using the
        training data.'''
        assert type(y[0]) == str, f'Classifier._fit: Input target values must be string labels, not {y.dtype}.'
        self.encoder.fit(y.reshape(-1, 1))
        self.loss_func.fit(y, categories=self.encoder.categories_[0])
        self.scaler.fit(X)

    def _preprocess(self, X:np.ndarray, y:np.ndarray=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''Encodes and transforms the input arrays, and converts them to FloatTensors on the specified device.'''
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y = self.encoder.transform(y.reshape(-1, 1))
            y = torch.FloatTensor(y).to(self.device)
        return X, y

    def _accuracy(self, X:torch.FloatTensor, y:torch.FloatTensor) -> float:
        '''Compute the balanced accuracy score based on model outputs and one-hot encoded targets.'''
        y_pred = self(X, one_hot=True).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        # Convert one-hot encoded y arrays to labels. 
        y_pred = self.encoder.inverse_transform(y_pred).ravel()
        y = self.encoder.inverse_transform(y).ravel()
        return balanced_accuracy_score(y, y_pred)

    def accuracy(self, X:np.ndarray, y:np.ndarray) -> float:
        '''Compute the balanced accuracy score based on model outputs and dataset labels (strings).'''
        X, _ = self._preprocess(X) # Standardize the input features. 
        y_pred = self(X, one_hot=True).cpu().detach().numpy()
        y_pred = self.encoder.inverse_transform(y_pred).ravel() # Convert one-hot encoded array to labels. 
        return balanced_accuracy_score(y, y_pred)


    def fit(self, X:np.ndarray, y:np.ndarray, X_val:np.ndarray, y_val:np.ndarray,
            lr:float=0.001, 
            batch_size:int=16,
            epochs:int=100):

        self._fit(X, y)
        # Standardize the X arrays and one-hot encode the labels. 

        X, y = self._preprocess(X, y)
        X_val, y_val = self._preprocess(X_val, y_val)

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=0.01) # Weight decay sets regularization strength. 
        best_epoch, best_model_weights = 0, copy.deepcopy(self.state_dict()) 

        self.train() # Classifier in train mode. 

        train_losses, val_accs = [], []
        pbar = tqdm(list(range(epochs)), desc='Classifier.fit: Training Classifier...')
        for epoch in pbar:
            X, y = Classifier.shuffle(X, y) # Shuffle the transformed data.
            batch_losses = [] 
            for X_batch, y_batch in zip(*Classifier.get_batches(X, y)):
                loss = self.loss_func(self(X_batch, one_hot=False), y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())      
            train_losses.append(np.mean(batch_losses)) # Store the average weighted train losses over the epoch. 
            val_accs.append(self._accuracy(X_val, y_val)) # Store model accuracy on the validation dataset. 

            if val_accs[-1] >= max(val_accs):
                best_epoch = epoch
                best_model_weights = copy.deepcopy(self.state_dict())

            pbar.set_description(f'Classifier.fit: Training Classifier... best validation accuracy {np.round(max(val_accs), 2)} encountered at epoch {best_epoch}.')
            

        self.load_state_dict(best_model_weights) # Load the best enountered model weights.
        print(f'Classifier.fit: Best validation accuracy of {np.round(max(val_accs), 2)} achieved at epoch {best_epoch}.')
        
        self.eval()
        
        return train_losses, val_accs, best_epoch

    def predict(self, X:np.ndarray) -> np.ndarray:
        '''Apply the model to input data, returning an array of string labels.'''
        X, _ = self._preprocess(X) # Don't forget to Z-score scale the data!
        y_pred = self(X, one_hot=True).cpu().detach().numpy() # Convert output tensor to numpy array. 
        return self.encoder.inverse_transform(y_pred).ravel()

    @classmethod
    def load(cls, path:str):
        with open(path, 'rb') as f:
            # obj = pickle.load(f)
            obj = Unpickler(f).load()
        return obj    


class TernaryClassifier(Classifier):

    def __init__(self, input_dim:int=None):
        
        super().__init__(input_dim=input_dim, output_dim=3)


class BinaryClassifier(Classifier):

    def __init__(self, input_dim:int=None):

        super().__init__(input_dim=input_dim, output_dim=2)
