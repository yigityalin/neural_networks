# rnn.py
# Solution for Question 3

from collections import defaultdict, namedtuple
from itertools import product
from pathlib import Path
import json

from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# IMPORTANT: Please change this according to your own local paths to run the code
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent  # The root directory of the project
DATA_DIR = PROJECT_ROOT_DIR / 'data'  # The directory that contains the data
LOGS_DIR = PROJECT_ROOT_DIR / 'logs'  # The directory that contains the logs
RNN_DATA_PATH = DATA_DIR / 'data3.h5'  # The filepath to the dataset for q1
RNN_LOGS_DIR = LOGS_DIR / 'rnn'  # The directory to which the logs will be saved for q1

LOGS_DIR.mkdir(exist_ok=True)
RNN_LOGS_DIR.mkdir(exist_ok=True)


def load_and_preprocess_data_q3(filepath=RNN_DATA_PATH):
    with h5py.File(filepath) as file:
        X_train = np.asarray(file.get('trX'), dtype=np.float64)
        y_train = np.asarray(file.get('trY'), dtype=np.int32)
        X_test = np.asarray(file.get('tstX'), dtype=np.float64)
        y_test = np.asarray(file.get('tstY'), dtype=np.int32)
    n = len(X_train)
    rng = np.random.default_rng()

    indices = np.arange(n)
    valid_indices = rng.choice(indices, n // 10, replace=False)
    train_indices = np.setdiff1d(indices, valid_indices)
    X_valid, y_valid = X_train[valid_indices], y_train[valid_indices]
    X_train, y_train = X_train[train_indices], y_train[train_indices]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def relu(z):
    return np.maximum(0, z)


def relu_backward(z):
    return np.where(z > 0, 1, 0)


def softmax(z):
    exp = np.exp(z - z.max())
    return exp / exp.sum(axis=0)


def categorical_cross_entropy(y_pred, y_true):
    """Calculates the categorical cross entropy loss"""
    masked = np.multiply(y_pred, y_true)
    logloss = np.where(masked != 0, np.log(masked, where=masked != 0), 0)
    return -np.sum(logloss, axis=1)


def accuracy_score(y_pred, y_true):
    labels = y_true.argmax(axis=1)
    preds = y_pred.argmax(axis=1)
    return np.mean(labels == preds)


# Weight and gradient containers for readability
RecurrentLayerWeights = namedtuple('RecurrentLayerWeights', ['b', 'Wx', 'Wh'])
RecurrentLayerGradients = RecurrentLayerWeights

FullyConnectedLayerWeights = namedtuple('FullyConnectedLayerWeights', ['b', 'W'])
FullyConnectedLayerGradients = FullyConnectedLayerWeights


class SimpleRNN:
    def __init__(self, recurrent_units, n_neurons):
        self.input_units = None                      # will be determined when while training
        self.recurrent_units = recurrent_units       # number of neurons in the recurrent layer
        self.n_neurons = n_neurons                   # number of neurons in the layers of MLP
        self.R = None                                # recurrent layer
        self.perceptron = None                       # MLP layers

    def initialize_layers(self):
        """Initializes the weights and the hidden state of the recurrent layer"""
        rng = np.random.default_rng()

        # initialize recurrent layer weights
        bound = np.sqrt(6 / (self.input_units + self.recurrent_units))
        b = rng.uniform(-bound, bound, size=(self.recurrent_units, 1))
        Wx = rng.uniform(-bound, bound, size=(self.recurrent_units, self.input_units))

        bound = np.sqrt(3 / self.recurrent_units)
        Wh = rng.uniform(-bound, bound, size=(self.recurrent_units, self.recurrent_units))

        self.R = RecurrentLayerWeights(b, Wx, Wh)

        # initialize MLP weights
        self.perceptron = []
        for i, n in enumerate(self.n_neurons):
            n_prev = self.n_neurons[i - 1] if i != 0 else self.recurrent_units
            bound = np.sqrt(6 / (n + n_prev))
            b = rng.uniform(-bound, bound, size=(n, 1))
            W = rng.uniform(-bound, bound, size=(n, n_prev))

            layer = FullyConnectedLayerWeights(b, W)
            self.perceptron.append(layer)

    def __call__(self, X):
        """Inference mode forward pass through the network"""
        X = np.transpose(X, axes=(1, 2, 0))

        H = np.zeros((self.recurrent_units, X.shape[-1]))
        Z = X.T
        for i, X_ in enumerate(X):
            V = self.R.Wh @ H + self.R.Wx @ X_ + self.R.b
            H = Z = np.tanh(V)

        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)

        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = softmax(V)
        return Z.T

    def forward(self, X, y):
        """
        Training mode forward pass though the network.
        Caches the pre- and post-activation outputs of the layers
        """
        R_Z_cache = []
        H = np.zeros((self.recurrent_units, X.shape[-1]))
        Z = X
        for i, X_ in enumerate(X):
            V = self.R.Wh @ H + self.R.Wx @ X_ + self.R.b
            H = Z = np.tanh(V)

            R_Z_cache.append(Z)

        V_cache, Z_cache = [], []
        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)

            V_cache.append(V)
            Z_cache.append(Z)

        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = softmax(V)

        V_cache.append(V)
        Z_cache.append(Z)

        # compute training metrics
        J = np.mean(categorical_cross_entropy(Z.T, y.T))  # compute cross entropy
        train_accuracy = accuracy_score(Z.T, y.T)         # compute accuracy
        return J, train_accuracy, R_Z_cache, V_cache, Z_cache

    def backward(self, X, y, R_Z_cache, V_cache, Z_cache, unfold):
        """Training mode backward pass through the network"""
        # calculate the gradients for the MLP
        gradients = []

        delta = Z_cache[-1] - y
        db = np.mean(delta, axis=1, keepdims=True)
        dW = delta @ Z_cache[-2].T / Z_cache[-2].shape[-1]
        gradients.append(FullyConnectedLayerGradients(db, dW))

        for i in reversed(range(1, len(self.perceptron) - 1)):
            delta = (self.perceptron[i + 1].W.T @ delta) * relu_backward(V_cache[i])
            db = np.mean(delta, axis=1, keepdims=True)
            dW = delta @ Z_cache[i - 1].T / Z_cache[i - 1].shape[-1]
            gradients.append(FullyConnectedLayerGradients(db, dW))

        delta = (self.perceptron[1].W.T @ delta) * relu_backward(V_cache[0])
        db = np.mean(delta, axis=1, keepdims=True)
        dW = (delta @ R_Z_cache[-1].T) / R_Z_cache[-1].shape[-1]
        gradients.append(FullyConnectedLayerGradients(db, dW))

        gradients.reverse()  # reverse the gradient list to get the correct order

        # calculate the gradients for the recurrent layer
        delta = (self.perceptron[0].W.T @ delta) * (1 - R_Z_cache[-1] ** 2)
        db = np.mean(delta, axis=1, keepdims=True)
        dWx = delta @ X[-1].T / self.recurrent_units
        dWh = delta @ R_Z_cache[-2].T / self.recurrent_units

        for i in reversed(range(len(X) - unfold, len(X) - 1)):
            delta = (self.R.Wh.T @ delta) * (1 - R_Z_cache[i] ** 2)
            db += np.mean(delta, axis=1, keepdims=True)
            dWx += delta @ X[i].T / self.recurrent_units
            dWh += delta @ R_Z_cache[i].T / self.recurrent_units

        dR = RecurrentLayerGradients(db, dWx, dWh)
        return dR, gradients

    def step(self, X, y, unfold):
        """Combines a forward and a backward pass through the network"""
        J, train_accuracy, R_Z_cache, V_cache, Z_cache = self.forward(X, y)
        dR, gradients = self.backward(X, y, R_Z_cache, V_cache, Z_cache, unfold)
        return J, train_accuracy, dR, gradients

    def fit(self,
            X,
            y,
            X_valid,
            y_valid,
            alpha=0.1,
            momentum=0.85,
            epochs=50,
            batch_size=32,
            unfold=150,
            tolerance=5,
            shuffle=True,
            cold_start=False
            ):
        """
        Train the neural network
        :param X: the training features
        :param y: the training labels
        :param alpha: the learning rate
        :param momentum: the momentum
        :param epochs: the number of training epochs
        :param batch_size: the size of the training batches
        :param unfold: number of time steps to backpropagate in time
        :param tolerance: the number of epochs without improvement for early stopping
        :param shuffle: whether to shuffle the data each epoch
        :param X_valid: the validation features
        :param y_valid: the validation labels
        :param cold_start: whether to reinitialize the weights before training
        :return:
        """
        if cold_start or self.R is None or self.perceptron is None:
            self.input_units = X.shape[-1]
            self.initialize_layers()

        n_batches = len(X) // batch_size
        history = defaultdict(list)
        delta_R_prev, delta_weights_prev = None, None

        for epoch in (progress_bar := tqdm(range(epochs))):
            # obtain the shuffled indices for training and validation
            train_indices = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
            batch_accuracies = []
            batch_avg_losses = []
            for batch in range(n_batches):
                # get the batch data using the shuffled indices
                batch_indices = train_indices[batch * batch_size: (batch + 1) * batch_size]
                X_batch = np.transpose(X[batch_indices], axes=(1, 2, 0))
                y_batch = np.transpose(y[batch_indices])

                # forward and backward passes through the network
                J, train_accuracy, dR, gradients = self.step(X_batch, y_batch, unfold)

                batch_avg_losses.append(J)
                batch_accuracies.append(train_accuracy)

                # calculate the updates given previous updates and gradients
                delta_R, delta_weights = self.calculate_updates(delta_R_prev, delta_weights_prev,
                                                                dR, gradients, momentum)
                self.apply_updates(delta_R, delta_weights, alpha)

                # save previous updates for momentum
                delta_R_prev = delta_R
                delta_weights_prev = delta_weights

            # test model performance on validation dataset
            y_valid_pred = self(X_valid)
            valid_avg_loss = np.mean(categorical_cross_entropy(y_valid_pred, y_valid))
            train_avg_loss = np.mean(batch_avg_losses)
            valid_accuracy = accuracy_score(y_valid_pred, y_valid)
            train_accuracy = np.mean(batch_accuracies)

            # log training and validation metrics
            history['train_cross_entropy'].append(train_avg_loss)
            history['train_accuracy'].append(train_accuracy)
            history['valid_cross_entropy'].append(valid_avg_loss)
            history['valid_accuracy'].append(valid_accuracy)

            progress_bar.set_description_str(f'n_neurons={"-".join([str(i) for i in self.n_neurons])}, '
                                             f'alpha={alpha}, momentum={momentum}')
            progress_bar.set_postfix_str(f'train_cross_entropy={train_avg_loss:.7f}, '
                                         f'train_accuracy={train_accuracy:.3f}, '
                                         f'valid_cross_entropy={valid_avg_loss:.7f}, '
                                         f'valid_accuracy={valid_accuracy:.3f}')

            # stop the training if there is no improvement in last "tolerance" episodes
            if epoch > 10 and np.argmin(history['valid_cross_entropy'][-tolerance:]) == 0:
                break
        return history

    @staticmethod
    def calculate_updates(delta_R_prev, delta_weights_prev, dR, gradients, momentum):
        """Calculate the weight updates with momentum given previous updates and gradients"""
        delta_R = RecurrentLayerWeights(*[
            momentum * delta_R_prev_ + (1 - momentum) * dR_
            for delta_R_prev_, dR_ in zip(delta_R_prev, dR)
        ]) if delta_R_prev else dR
        delta_weights = [
            FullyConnectedLayerWeights(*[momentum * delta_w_prev + (1 - momentum) * grads
                                         for delta_w_prev, grads in zip(delta_weights_prev_, gradients_)])
            for delta_weights_prev_, gradients_ in zip(delta_weights_prev, gradients)
        ] if delta_weights_prev else gradients

        return delta_R, delta_weights

    def apply_updates(self, delta_R, delta_weights, alpha):
        """Apply the calculated updates to the network weights"""
        self.R = RecurrentLayerWeights(*[w - alpha * delta for w, delta in zip(self.R, delta_R)])
        self.perceptron = [
            FullyConnectedLayerWeights(*[w - alpha * delta for w, delta in zip(layer, delta_weights_)])
            for layer, delta_weights_ in zip(self.perceptron, delta_weights)
        ]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LSTM:
    GATES = ['i', 'f', 'c', 'o']

    def __init__(self, recurrent_units, n_neurons):
        self.input_units = None  # will be determined when while training
        self.recurrent_units = recurrent_units  # number of neurons in the recurrent layer
        self.n_neurons = n_neurons  # number of neurons in the layers of MLP
        self.R = None  # recurrent layer
        self.perceptron = None  # MLP layers

    def initialize_layers(self):
        rng = np.random.default_rng()

        # initialize MLP weights
        self.perceptron = []
        for i, n in enumerate(self.n_neurons):
            n_prev = self.n_neurons[i - 1] if i != 0 else self.recurrent_units
            bound = np.sqrt(6 / (n + n_prev))
            b = rng.uniform(-bound, bound, size=(n, 1))
            W = rng.uniform(-bound, bound, size=(n, n_prev))

            layer = FullyConnectedLayerWeights(b, W)
            self.perceptron.append(layer)

        bound_in = np.sqrt(6 / (self.input_units + self.recurrent_units))
        bound_h = np.sqrt(3 / self.recurrent_units)
        self.R = {
            gate: {'b': rng.uniform(-bound_in, bound_in, size=(self.recurrent_units, 1)),
                   'Wx': rng.uniform(-bound_in, bound_in, size=(self.recurrent_units, self.input_units)),
                   'Wh': rng.uniform(-bound_h, bound_h, size=(self.recurrent_units, self.recurrent_units))}
            for gate in self.GATES
        }

    def __call__(self, X):
        X = np.transpose(X, axes=(1, 2, 0))

        H_prev = np.zeros((self.recurrent_units, X.shape[-1]))  # hidden state
        C_prev = np.zeros((self.recurrent_units, X.shape[-1]))  # cell state
        for i, Z in enumerate(X):
            # input gate
            Vi = self.R['i']['Wx'] @ Z + self.R['i']['Wh'] @ H_prev + self.R['i']['b']
            Zi = sigmoid(Vi)

            # forget gate
            Vf = self.R['f']['Wx'] @ Z + self.R['f']['Wh'] @ H_prev + self.R['f']['b']
            Zf = sigmoid(Vf)

            # cell state
            Vc = self.R['c']['Wx'] @ Z + self.R['c']['Wh'] @ H_prev + self.R['c']['b']
            Zc = np.tanh(Vc)

            # output gate
            Vo = self.R['o']['Wx'] @ Z + self.R['o']['Wh'] @ H_prev + self.R['o']['b']
            Zo = sigmoid(Vo)

            C = C_prev * Zf + Zc * Zi
            H = Zo * np.tanh(C)

            C_prev = C.copy()
            H_prev = H.copy()

        Z = H
        # MLP forward
        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)

        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = softmax(V)

        return Z.T

    def forward(self, X, y):
        R_Z_cache, C_cache, H_cache = [], [], []
        H_prev = np.zeros((self.recurrent_units, X.shape[-1]))  # hidden state
        C_prev = np.zeros((self.recurrent_units, X.shape[-1]))  # cell state
        for i, Z in enumerate(X):
            # input gate
            Vi = self.R['i']['Wx'] @ Z + self.R['i']['Wh'] @ H_prev + self.R['i']['b']
            Zi = sigmoid(Vi)

            # forget gate
            Vf = self.R['f']['Wx'] @ Z + self.R['f']['Wh'] @ H_prev + self.R['f']['b']
            Zf = sigmoid(Vf)

            # cell state
            Vc = self.R['c']['Wx'] @ Z + self.R['c']['Wh'] @ H_prev + self.R['c']['b']
            Zc = np.tanh(Vc)

            # output gate
            Vo = self.R['o']['Wx'] @ Z + self.R['o']['Wh'] @ H_prev + self.R['o']['b']
            Zo = sigmoid(Vo)

            C = C_prev * Zf + Zc * Zi
            H = Zo * np.tanh(C)

            R_Z_cache.append({'i': Zi.copy(), 'f': Zf.copy(), 'c': Zc.copy(), 'o': Zo.copy()})
            C_cache.append(C.copy())
            H_cache.append(H.copy())

            C_prev = C.copy()
            H_prev = H.copy()

        Z = H
        # MLP forward
        V_cache, Z_cache = [], []
        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)

            V_cache.append(V)
            Z_cache.append(Z)

        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = softmax(V)

        V_cache.append(V)
        Z_cache.append(Z)

        J = np.mean(categorical_cross_entropy(Z.T, y.T))  # compute cross entropy
        train_accuracy = accuracy_score(Z.T, y.T)  # compute accuracy
        return J, train_accuracy, R_Z_cache, C_cache, H_cache, V_cache, Z_cache

    def backward(self, X, y, R_Z_cache, C_cache, H_cache, V_cache, Z_cache, unfold):
        # calculate the gradients for the MLP
        mlp_gradients = []

        delta = Z_cache[-1] - y
        db = np.mean(delta, axis=1, keepdims=True)
        dW = delta @ Z_cache[-2].T / Z_cache[-2].shape[-1]
        mlp_gradients.append(FullyConnectedLayerGradients(db, dW))

        for j in reversed(range(1, len(self.perceptron) - 1)):
            delta = (self.perceptron[j + 1].W.T @ delta) * relu_backward(V_cache[j])
            db = np.mean(delta, axis=1, keepdims=True)
            dW = delta @ Z_cache[j - 1].T / Z_cache[j - 1].shape[-1]
            mlp_gradients.append(FullyConnectedLayerGradients(db, dW))

        delta = (self.perceptron[1].W.T @ delta) * relu_backward(V_cache[0])
        db = np.mean(delta, axis=1, keepdims=True)
        dW = (delta @ H_cache[-1].T) / H_cache[-1].shape[-1]
        mlp_gradients.append(FullyConnectedLayerGradients(db, dW))

        mlp_gradients.reverse()  # reverse the gradient list to get the correct order

        # calculate the gradients for the recurrent layer
        delta = self.perceptron[0].W.T @ delta

        lstm_gradients = {
            gate: {name: np.zeros_like(W) for name, W in weights.items()}
            for gate, weights in self.R.items()
        }

        deltas = {
            'i': delta * R_Z_cache[-1]['o'] * (1 - np.tanh(C_cache[-1]) ** 2) * R_Z_cache[-1]['c'] * R_Z_cache[-1]['i'] * (1 - R_Z_cache[-1]['i']),
            'f': delta * R_Z_cache[-1]['o'] * (1 - np.tanh(C_cache[-1]) ** 2) * C_cache[-2] * R_Z_cache[-1]['f'] * (1 - R_Z_cache[-1]['f']),
            'c': delta * R_Z_cache[-1]['o'] * (1 - np.tanh(C_cache[-1]) ** 2) * R_Z_cache[-1]['i'] * (1 - R_Z_cache[-1]['c'] ** 2),
            'o': delta * np.tanh(C_cache[-1]) * R_Z_cache[-1]['o'] * (1 - R_Z_cache[-1]['o']),
        }

        for gate in lstm_gradients.keys():
            lstm_gradients[gate]['b'] += np.mean(deltas[gate], axis=1, keepdims=True)
            lstm_gradients[gate]['Wx'] += deltas[gate] @ X[-1].T / self.recurrent_units
            lstm_gradients[gate]['Wh'] += deltas[gate] @ H_cache[-2].T / self.recurrent_units

        for t in reversed(range(len(X) - unfold, len(X) - 1)):
            if t == 0:
                break

            deltas['i'] = (self.R['i']['Wh'].T @ deltas['i']) * R_Z_cache[t]['o'] * (1 - np.tanh(C_cache[t]) ** 2) * R_Z_cache[t]['c'] * R_Z_cache[t]['i'] * (1 - R_Z_cache[t]['i'])
            deltas['f'] = (self.R['f']['Wh'].T @ deltas['f']) * R_Z_cache[t]['o'] * (1 - np.tanh(C_cache[t]) ** 2) * C_cache[t - 1] * R_Z_cache[t]['f'] * (1 - R_Z_cache[t]['f'])
            deltas['c'] = (self.R['c']['Wh'].T @ deltas['c']) * R_Z_cache[t]['o'] * (1 - np.tanh(C_cache[t]) ** 2) * R_Z_cache[t]['i'] * (1 - R_Z_cache[t]['c'] ** 2)
            deltas['o'] = (self.R['o']['Wh'].T @ deltas['o']) * np.tanh(C_cache[-1]) * R_Z_cache[-1]['o'] * (1 - R_Z_cache[-1]['o'])

            for gate in lstm_gradients.keys():
                lstm_gradients[gate]['b'] += np.mean(deltas[gate], axis=1, keepdims=True)
                lstm_gradients[gate]['Wx'] += deltas[gate] @ X[t].T / self.recurrent_units
                lstm_gradients[gate]['Wh'] += deltas[gate] @ H_cache[t - 1].T / self.recurrent_units

        return lstm_gradients, mlp_gradients

    def step(self, X, y, unfold):
        """Combines a forward and a backward pass through the network"""
        J, train_accuracy, R_Z_cache, C_cache, H_cache, V_cache, Z_cache = self.forward(X, y)
        lstm_gradients, mlp_gradients = self.backward(X, y, R_Z_cache, C_cache,H_cache, V_cache, Z_cache, unfold)
        return J, train_accuracy, lstm_gradients, mlp_gradients

    def fit(self,
            X,
            y,
            X_valid,
            y_valid,
            alpha=0.1,
            momentum=0.85,
            epochs=50,
            batch_size=32,
            unfold=150,
            tolerance=5,
            shuffle=True,
            cold_start=False
            ):
        """
        Train the neural network
        :param X: the training features
        :param y: the training labels
        :param alpha: the learning rate
        :param momentum: the momentum
        :param epochs: the number of training epochs
        :param batch_size: the size of the training batches
        :param unfold: number of time steps to backpropagate in time
        :param tolerance: the number of epochs without improvement for early stopping
        :param shuffle: whether to shuffle the data each epoch
        :param X_valid: the validation features
        :param y_valid: the validation labels
        :param cold_start: whether to reinitialize the weights before training
        :return:
        """
        if cold_start or self.R is None or self.perceptron is None:
            self.input_units = X.shape[-1]
            self.initialize_layers()

        n_batches = len(X) // batch_size
        history = defaultdict(list)
        delta_R_prev, delta_weights_prev = None, None

        for epoch in (progress_bar := tqdm(range(epochs))):
            # obtain the shuffled indices for training and validation
            train_indices = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
            batch_accuracies = []
            batch_avg_losses = []
            for batch in range(n_batches):
                # get the batch data using the shuffled indices
                batch_indices = train_indices[batch * batch_size: (batch + 1) * batch_size]
                X_batch = np.transpose(X[batch_indices], axes=(1, 2, 0))
                y_batch = np.transpose(y[batch_indices])

                # forward and backward passes through the network
                J, train_accuracy, lstm_gradients, mlp_gradients = self.step(X_batch, y_batch, unfold)

                batch_avg_losses.append(J)
                batch_accuracies.append(train_accuracy)

                # calculate the updates given previous updates and gradients
                delta_R, delta_weights = self.calculate_updates(delta_R_prev, delta_weights_prev,
                                                                lstm_gradients, mlp_gradients, momentum)
                self.apply_updates(delta_R, delta_weights, alpha)

                # save previous updates for momentum
                delta_R_prev = delta_R
                delta_weights_prev = delta_weights

            # test model performance on validation dataset
            y_valid_pred = self(X_valid)
            valid_avg_loss = np.mean(categorical_cross_entropy(y_valid_pred, y_valid))
            train_avg_loss = np.mean(batch_avg_losses)
            valid_accuracy = accuracy_score(y_valid_pred, y_valid)
            train_accuracy = np.mean(batch_accuracies)

            # log training and validation metrics
            history['train_cross_entropy'].append(train_avg_loss)
            history['train_accuracy'].append(train_accuracy)
            history['valid_cross_entropy'].append(valid_avg_loss)
            history['valid_accuracy'].append(valid_accuracy)

            progress_bar.set_description_str(f'n_neurons={"-".join([str(i) for i in self.n_neurons])}, '
                                             f'alpha={alpha}, momentum={momentum}')
            progress_bar.set_postfix_str(f'train_cross_entropy={train_avg_loss:.7f}, '
                                         f'train_accuracy={train_accuracy:.3f}, '
                                         f'valid_cross_entropy={valid_avg_loss:.7f}, '
                                         f'valid_accuracy={valid_accuracy:.3f}')

            # stop the training if there is no improvement in last "tolerance" episodes
            if epoch > 10 and np.argmin(history['valid_cross_entropy'][-tolerance:]) == 0:
                break
        return history

    @staticmethod
    def calculate_updates(delta_R_prev, delta_weights_prev, lstm_gradients, mlp_gradients, momentum):
        # lstm updates
        delta_R = {
            gate: {name: momentum * delta_W_prev + (1 - momentum) * delta_W
                   for delta_W_prev, (name, delta_W) in zip(delta_gate_prev.values(), gate_gradients.items())}
            for delta_gate_prev, (gate, gate_gradients) in zip(delta_R_prev.values(), lstm_gradients.items())
        } if delta_R_prev else lstm_gradients

        # mlp updates
        delta_weights = [
            FullyConnectedLayerWeights(*[momentum * delta_w_prev + (1 - momentum) * grads
                                         for delta_w_prev, grads in zip(delta_weights_prev_, gradients_)])
            for delta_weights_prev_, gradients_ in zip(delta_weights_prev, mlp_gradients)
        ] if delta_weights_prev else mlp_gradients

        return delta_R, delta_weights

    def apply_updates(self, delta_R, delta_weights, alpha):
        for gate, gate_updates in delta_R.items():
            for name, W_update in gate_updates.items():
                self.R[gate][name] -= W_update
        self.perceptron = [
            FullyConnectedLayerWeights(*[w - alpha * delta for w, delta in zip(layer, delta_weights_)])
            for layer, delta_weights_ in zip(self.perceptron, delta_weights)
        ]


class GRU:
    GATES = ['i', 'c', 'o']

    def __init__(self, recurrent_units, n_neurons):
        self.input_units = None  # will be determined when while training
        self.recurrent_units = recurrent_units  # number of neurons in the recurrent layer
        self.n_neurons = n_neurons  # number of neurons in the layers of MLP
        self.R = None  # recurrent layer
        self.perceptron = None  # MLP layers

    def initialize_layers(self):
        rng = np.random.default_rng()

        # initialize MLP weights
        self.perceptron = []
        for i, n in enumerate(self.n_neurons):
            n_prev = self.n_neurons[i - 1] if i != 0 else self.recurrent_units
            bound = np.sqrt(6 / (n + n_prev))
            b = rng.uniform(-bound, bound, size=(n, 1))
            W = rng.uniform(-bound, bound, size=(n, n_prev))

            layer = FullyConnectedLayerWeights(b, W)
            self.perceptron.append(layer)

        bound_in = np.sqrt(6 / (self.input_units + self.recurrent_units))
        bound_h = np.sqrt(3 / self.recurrent_units)
        self.R = {
            gate: {'b': rng.uniform(-bound_in, bound_in, size=(self.recurrent_units, 1)),
                   'Wx': rng.uniform(-bound_in, bound_in, size=(self.recurrent_units, self.input_units)),
                   'Wh': rng.uniform(-bound_h, bound_h, size=(self.recurrent_units, self.recurrent_units))}
            for gate in self.GATES
        }

    def __call__(self, X):
        X = np.transpose(X, axes=(1, 2, 0))

        H_prev = np.zeros((self.recurrent_units, X.shape[-1]))  # hidden state
        C_prev = np.zeros((self.recurrent_units, X.shape[-1]))  # cell state
        for i, Z in enumerate(X):
            # input gate
            Vi = self.R['i']['Wx'] @ Z + self.R['i']['Wh'] @ H_prev + self.R['i']['b']
            Zi = sigmoid(Vi)

            # forget gate
            Zf = 1 - Zi

            # cell state
            Vc = self.R['c']['Wx'] @ Z + self.R['c']['Wh'] @ H_prev + self.R['c']['b']
            Zc = np.tanh(Vc)

            # output gate
            Vo = self.R['o']['Wx'] @ Z + self.R['o']['Wh'] @ H_prev + self.R['o']['b']
            Zo = sigmoid(Vo)

            C = C_prev * Zf + Zc * Zi
            H = Zo * np.tanh(C)

            C_prev = C.copy()
            H_prev = H.copy()

        Z = H
        # MLP forward
        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)

        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = softmax(V)

        return Z.T

    def forward(self, X, y):
        R_Z_cache, C_cache, H_cache = [], [], []
        H_prev = np.zeros((self.recurrent_units, X.shape[-1]))  # hidden state
        C_prev = np.zeros((self.recurrent_units, X.shape[-1]))  # cell state
        for i, Z in enumerate(X):
            # input gate
            Vi = self.R['i']['Wx'] @ Z + self.R['i']['Wh'] @ H_prev + self.R['i']['b']
            Zi = sigmoid(Vi)

            # forget gate
            Zf = 1 - Zi

            # cell state
            Vc = self.R['c']['Wx'] @ Z + self.R['c']['Wh'] @ H_prev + self.R['c']['b']
            Zc = np.tanh(Vc)

            # output gate
            Vo = self.R['o']['Wx'] @ Z + self.R['o']['Wh'] @ H_prev + self.R['o']['b']
            Zo = sigmoid(Vo)

            C = C_prev * Zf + Zc * Zi
            H = Zo * np.tanh(C)

            R_Z_cache.append({'i': Zi.copy(), 'f': Zf.copy(), 'c': Zc.copy(), 'o': Zo.copy()})
            C_cache.append(C.copy())
            H_cache.append(H.copy())

            C_prev = C.copy()
            H_prev = H.copy()

        Z = H
        # MLP forward
        V_cache, Z_cache = [], []
        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)

            V_cache.append(V)
            Z_cache.append(Z)

        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = softmax(V)

        V_cache.append(V)
        Z_cache.append(Z)

        J = np.mean(categorical_cross_entropy(Z.T, y.T))  # compute cross entropy
        train_accuracy = accuracy_score(Z.T, y.T)  # compute accuracy
        return J, train_accuracy, R_Z_cache, C_cache, H_cache, V_cache, Z_cache

    def backward(self, X, y, R_Z_cache, C_cache, H_cache, V_cache, Z_cache, unfold):
        # calculate the gradients for the MLP
        mlp_gradients = []

        delta = Z_cache[-1] - y
        db = np.mean(delta, axis=1, keepdims=True)
        dW = delta @ Z_cache[-2].T / Z_cache[-2].shape[-1]
        mlp_gradients.append(FullyConnectedLayerGradients(db, dW))

        for j in reversed(range(1, len(self.perceptron) - 1)):
            delta = (self.perceptron[j + 1].W.T @ delta) * relu_backward(V_cache[j])
            db = np.mean(delta, axis=1, keepdims=True)
            dW = delta @ Z_cache[j - 1].T / Z_cache[j - 1].shape[-1]
            mlp_gradients.append(FullyConnectedLayerGradients(db, dW))

        delta = (self.perceptron[1].W.T @ delta) * relu_backward(V_cache[0])
        db = np.mean(delta, axis=1, keepdims=True)
        dW = (delta @ H_cache[-1].T) / H_cache[-1].shape[-1]
        mlp_gradients.append(FullyConnectedLayerGradients(db, dW))

        mlp_gradients.reverse()  # reverse the gradient list to get the correct order

        # calculate the gradients for the recurrent layer
        delta = self.perceptron[0].W.T @ delta

        lstm_gradients = {
            gate: {name: np.zeros_like(W) for name, W in weights.items()}
            for gate, weights in self.R.items()
        }

        deltas = {
            'i': delta * R_Z_cache[-1]['o'] * (1 - np.tanh(C_cache[-1]) ** 2) * R_Z_cache[-1]['c'] * R_Z_cache[-1]['i'] * (1 - R_Z_cache[-1]['i']),
            'c': delta * R_Z_cache[-1]['o'] * (1 - np.tanh(C_cache[-1]) ** 2) * R_Z_cache[-1]['i'] * (1 - R_Z_cache[-1]['c'] ** 2),
            'o': delta * np.tanh(C_cache[-1]) * R_Z_cache[-1]['o'] * (1 - R_Z_cache[-1]['o']),
        }

        for gate in lstm_gradients.keys():
            lstm_gradients[gate]['b'] += np.mean(deltas[gate], axis=1, keepdims=True)
            lstm_gradients[gate]['Wx'] += deltas[gate] @ X[-1].T / self.recurrent_units
            lstm_gradients[gate]['Wh'] += deltas[gate] @ H_cache[-2].T / self.recurrent_units

        for t in reversed(range(len(X) - unfold, len(X) - 1)):
            if t == 0:
                break

            deltas['i'] = (self.R['i']['Wh'].T @ deltas['i']) * R_Z_cache[t]['o'] * (1 - np.tanh(C_cache[t]) ** 2) * R_Z_cache[t]['c'] * R_Z_cache[t]['i'] * (1 - R_Z_cache[t]['i'])
            deltas['c'] = (self.R['c']['Wh'].T @ deltas['c']) * R_Z_cache[t]['o'] * (1 - np.tanh(C_cache[t]) ** 2) * R_Z_cache[t]['i'] * (1 - R_Z_cache[t]['c'] ** 2)
            deltas['o'] = (self.R['o']['Wh'].T @ deltas['o']) * np.tanh(C_cache[-1]) * R_Z_cache[-1]['o'] * (1 - R_Z_cache[-1]['o'])

            for gate in lstm_gradients.keys():
                lstm_gradients[gate]['b'] += np.mean(deltas[gate], axis=1, keepdims=True)
                lstm_gradients[gate]['Wx'] += deltas[gate] @ X[t].T / self.recurrent_units
                lstm_gradients[gate]['Wh'] += deltas[gate] @ H_cache[t - 1].T / self.recurrent_units

        return lstm_gradients, mlp_gradients

    def step(self, X, y, unfold):
        """Combines a forward and a backward pass through the network"""
        J, train_accuracy, R_Z_cache, C_cache, H_cache, V_cache, Z_cache = self.forward(X, y)
        lstm_gradients, mlp_gradients = self.backward(X, y, R_Z_cache, C_cache,H_cache, V_cache, Z_cache, unfold)
        return J, train_accuracy, lstm_gradients, mlp_gradients

    def fit(self,
            X,
            y,
            X_valid,
            y_valid,
            alpha=0.1,
            momentum=0.85,
            epochs=50,
            batch_size=32,
            unfold=150,
            tolerance=5,
            shuffle=True,
            cold_start=False
            ):
        """
        Train the neural network
        :param X: the training features
        :param y: the training labels
        :param alpha: the learning rate
        :param momentum: the momentum
        :param epochs: the number of training epochs
        :param batch_size: the size of the training batches
        :param unfold: number of time steps to backpropagate in time
        :param tolerance: the number of epochs without improvement for early stopping
        :param shuffle: whether to shuffle the data each epoch
        :param X_valid: the validation features
        :param y_valid: the validation labels
        :param cold_start: whether to reinitialize the weights before training
        :return:
        """
        if cold_start or self.R is None or self.perceptron is None:
            self.input_units = X.shape[-1]
            self.initialize_layers()

        n_batches = len(X) // batch_size
        history = defaultdict(list)
        delta_R_prev, delta_weights_prev = None, None

        for epoch in (progress_bar := tqdm(range(epochs))):
            # obtain the shuffled indices for training and validation
            train_indices = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
            batch_accuracies = []
            batch_avg_losses = []
            for batch in range(n_batches):
                # get the batch data using the shuffled indices
                batch_indices = train_indices[batch * batch_size: (batch + 1) * batch_size]
                X_batch = np.transpose(X[batch_indices], axes=(1, 2, 0))
                y_batch = np.transpose(y[batch_indices])

                # forward and backward passes through the network
                J, train_accuracy, lstm_gradients, mlp_gradients = self.step(X_batch, y_batch, unfold)

                batch_avg_losses.append(J)
                batch_accuracies.append(train_accuracy)

                # calculate the updates given previous updates and gradients
                delta_R, delta_weights = self.calculate_updates(delta_R_prev, delta_weights_prev,
                                                                lstm_gradients, mlp_gradients, momentum)
                self.apply_updates(delta_R, delta_weights, alpha)

                # save previous updates for momentum
                delta_R_prev = delta_R
                delta_weights_prev = delta_weights

            # test model performance on validation dataset
            y_valid_pred = self(X_valid)
            valid_avg_loss = np.mean(categorical_cross_entropy(y_valid_pred, y_valid))
            train_avg_loss = np.mean(batch_avg_losses)
            valid_accuracy = accuracy_score(y_valid_pred, y_valid)
            train_accuracy = np.mean(batch_accuracies)

            # log training and validation metrics
            history['train_cross_entropy'].append(train_avg_loss)
            history['train_accuracy'].append(train_accuracy)
            history['valid_cross_entropy'].append(valid_avg_loss)
            history['valid_accuracy'].append(valid_accuracy)

            progress_bar.set_description_str(f'n_neurons={"-".join([str(i) for i in self.n_neurons])}, '
                                             f'alpha={alpha}, momentum={momentum}')
            progress_bar.set_postfix_str(f'train_cross_entropy={train_avg_loss:.7f}, '
                                         f'train_accuracy={train_accuracy:.3f}, '
                                         f'valid_cross_entropy={valid_avg_loss:.7f}, '
                                         f'valid_accuracy={valid_accuracy:.3f}')

            # stop the training if there is no improvement in last "tolerance" episodes
            if epoch > 10 and np.argmin(history['valid_cross_entropy'][-tolerance:]) == 0:
                break
        return history

    @staticmethod
    def calculate_updates(delta_R_prev, delta_weights_prev, lstm_gradients, mlp_gradients, momentum):
        # lstm updates
        delta_R = {
            gate: {name: momentum * delta_W_prev + (1 - momentum) * delta_W
                   for delta_W_prev, (name, delta_W) in zip(delta_gate_prev.values(), gate_gradients.items())}
            for delta_gate_prev, (gate, gate_gradients) in zip(delta_R_prev.values(), lstm_gradients.items())
        } if delta_R_prev else lstm_gradients

        # mlp updates
        delta_weights = [
            FullyConnectedLayerWeights(*[momentum * delta_w_prev + (1 - momentum) * grads
                                         for delta_w_prev, grads in zip(delta_weights_prev_, gradients_)])
            for delta_weights_prev_, gradients_ in zip(delta_weights_prev, mlp_gradients)
        ] if delta_weights_prev else mlp_gradients

        return delta_R, delta_weights

    def apply_updates(self, delta_R, delta_weights, alpha):
        for gate, gate_updates in delta_R.items():
            for name, W_update in gate_updates.items():
                self.R[gate][name] -= W_update
        self.perceptron = [
            FullyConnectedLayerWeights(*[w - alpha * delta for w, delta in zip(layer, delta_weights_)])
            for layer, delta_weights_ in zip(self.perceptron, delta_weights)
        ]


def q3_main_part_a(X_train, y_train, X_valid, y_valid, X_test, y_test):
    part_a_logs_dir = RNN_LOGS_DIR / 'simple'
    part_a_logs_dir.mkdir(exist_ok=True)

    alphas = [0.01, 0.04, 0.08, 0.12]
    momentums = [0.85, 0.95]
    n_neurons_list = [[128, 128, 64, 6],
                      [128, 64, 32, 6],
                      [64, 64, 32, 6],
                      [32, 32, 6]]

    for alpha, momentum, n_neurons in product(alphas, momentums, n_neurons_list):
        model = SimpleRNN(recurrent_units=128, n_neurons=n_neurons)
        history = model.fit(X_train, y_train, X_valid, y_valid, alpha=alpha, momentum=momentum)

        train_preds = model(X_train)
        test_preds = model(X_test)

        history_filepath = part_a_logs_dir / f'model-part-a-{"_".join([str(i) for i in model.n_neurons])}' \
                                             f'alpha_{alpha}-momentum={momentum}-history'
        train_predictions_filepath = part_a_logs_dir / f'model-part-a-{"_".join([str(i) for i in model.n_neurons])}' \
                                                       f'alpha_{alpha}-momentum={momentum}-train_predictions'
        test_predictions_filepath = part_a_logs_dir / f'model-part-a-{"_".join([str(i) for i in model.n_neurons])}' \
                                                      f'alpha_{alpha}-momentum={momentum}-test_predictions'

        with open(history_filepath, 'w') as f:
            json.dump(history, f)

        with open(train_predictions_filepath, 'wb') as f:
            np.save(f, train_preds)

        with open(test_predictions_filepath, 'wb') as f:
            np.save(f, test_preds)


def q3_main_part_b(X_train, y_train, X_valid, y_valid, X_test, y_test):
    part_b_logs_dir = RNN_LOGS_DIR / 'lstm'
    part_b_logs_dir.mkdir(exist_ok=True)

    alphas = [0.01, 0.04, 0.08, 0.12]
    momentums = [0.85, 0.95]
    n_neurons_list = [[128, 128, 64, 6],
                      [128, 64, 32, 6],
                      [64, 64, 32, 6],
                      [32, 32, 6]]

    for alpha, momentum, n_neurons in product(alphas, momentums, n_neurons_list):
        model = LSTM(recurrent_units=128, n_neurons=n_neurons)
        history = model.fit(X_train, y_train, X_valid, y_valid, alpha=alpha, momentum=momentum)

        train_preds = model(X_train)
        test_preds = model(X_test)

        history_filepath = part_b_logs_dir / f'model-part-b-{"_".join([str(i) for i in model.n_neurons])}' \
                                             f'alpha_{alpha}-momentum={momentum}-history'
        train_predictions_filepath = part_b_logs_dir / f'model-part-b-{"_".join([str(i) for i in model.n_neurons])}' \
                                                       f'alpha_{alpha}-momentum={momentum}-train_predictions'
        test_predictions_filepath = part_b_logs_dir / f'model-part-b-{"_".join([str(i) for i in model.n_neurons])}' \
                                                      f'alpha_{alpha}-momentum={momentum}-test_predictions'

        with open(history_filepath, 'w') as f:
            json.dump(history, f)

        with open(train_predictions_filepath, 'wb') as f:
            np.save(f, train_preds)

        with open(test_predictions_filepath, 'wb') as f:
            np.save(f, test_preds)


def q3_main_part_c(X_train, y_train, X_valid, y_valid, X_test, y_test):
    part_c_logs_dir = RNN_LOGS_DIR / 'gru'
    part_c_logs_dir.mkdir(exist_ok=True)

    alphas = [0.01, 0.04, 0.08, 0.12]
    momentums = [0.85, 0.95]
    n_neurons_list = [[128, 128, 64, 6],
                      [128, 64, 32, 6],
                      [64, 64, 32, 6],
                      [32, 32, 6]]

    for alpha, momentum, n_neurons in product(alphas, momentums, n_neurons_list):
        model = GRU(recurrent_units=128, n_neurons=n_neurons)
        history = model.fit(X_train, y_train, X_valid, y_valid, alpha=alpha, momentum=momentum, tolerance=5)

        train_preds = model(X_train)
        test_preds = model(X_test)

        history_filepath = part_c_logs_dir / f'model-part-c-{"_".join([str(i) for i in model.n_neurons])}' \
                                             f'alpha_{alpha}-momentum={momentum}-history'
        train_predictions_filepath = part_c_logs_dir / f'model-part-c-{"_".join([str(i) for i in model.n_neurons])}' \
                                                       f'alpha_{alpha}-momentum={momentum}-train_predictions'
        test_predictions_filepath = part_c_logs_dir / f'model-part-c-{"_".join([str(i) for i in model.n_neurons])}' \
                                                      f'alpha_{alpha}-momentum={momentum}-test_predictions'

        with open(history_filepath, 'w') as f:
            json.dump(history, f)

        with open(train_predictions_filepath, 'wb') as f:
            np.save(f, train_preds)

        with open(test_predictions_filepath, 'wb') as f:
            np.save(f, test_preds)


def load_history(filepath):
    with open(filepath, 'r') as file:
        history = json.load(file)
    return history

def load_predictions(filepath):
    return np.load(filepath)


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data_q3()
    q3_main_part_a(X_train, y_train, X_valid, y_valid, X_test, y_test)
    q3_main_part_b(X_train, y_train, X_valid, y_valid, X_test, y_test)
    q3_main_part_c(X_train, y_train, X_valid, y_valid, X_test, y_test)

    alphas = [0.01, 0.04, 0.08, 0.12]
    momentums = [0.85, 0.95]
    n_neurons_list = [[128, 128, 64, 6],
                      [128, 64, 32, 6],
                      [64, 64, 32, 6],
                      [32, 32, 6]]
    hyperparameters = list(product(alphas, momentums, n_neurons_list))
    part_a_histories = [
        load_history(RNN_LOGS_DIR / 'simple' / f'model-part-a-{"_".join([str(i) for i in n_neurons])}' \
                                               f'alpha_{alpha}-momentum={momentum}-history')
        for alpha, momentum, n_neurons in hyperparameters
    ]
    part_b_histories = [
        load_history(RNN_LOGS_DIR / 'lstm' / f'model-part-b-{"_".join([str(i) for i in n_neurons])}' \
                                             f'alpha_{alpha}-momentum={momentum}-history')
        for alpha, momentum, n_neurons in hyperparameters
    ]
    part_c_histories = [
        load_history(RNN_LOGS_DIR / 'gru' / f'model-part-c-{"_".join([str(i) for i in n_neurons])}' \
                                            f'alpha_{alpha}-momentum={momentum}-history')
        for alpha, momentum, n_neurons in hyperparameters
    ]
    part_a_valid_accuracies = [history['valid_accuracy'][-1] for history in part_a_histories]
    part_b_valid_accuracies = [history['valid_accuracy'][-1] for history in part_b_histories]
    part_c_valid_accuracies = [history['valid_accuracy'][-1] for history in part_c_histories]

    part_a_best_index = np.argmax(part_a_valid_accuracies)
    part_b_best_index = np.argmax(part_b_valid_accuracies)
    part_c_best_index = np.argmax(part_c_valid_accuracies)

    part_a_best_params = hyperparameters[part_a_best_index]
    part_b_best_params = hyperparameters[part_b_best_index]
    part_c_best_params = hyperparameters[part_c_best_index]

    part_a_best_history = part_a_histories[part_a_best_index]
    part_b_best_history = part_b_histories[part_b_best_index]
    part_c_best_history = part_c_histories[part_c_best_index]

    alpha, momentum, n_neurons = part_a_best_params
    part_a_train_preds = load_predictions(
        RNN_LOGS_DIR / 'simple' / f'model-part-a-{"_".join([str(i) for i in n_neurons])}'\
                                  f'alpha_{alpha}-momentum={momentum}-train_predictions'
    )

    alpha, momentum, n_neurons = part_b_best_params
    part_b_train_preds = load_predictions(
        RNN_LOGS_DIR / 'lstm' / f'model-part-b-{"_".join([str(i) for i in n_neurons])}'\
                                  f'alpha_{alpha}-momentum={momentum}-train_predictions'
    )

    alpha, momentum, n_neurons = part_c_best_params
    part_c_train_preds = load_predictions(
        RNN_LOGS_DIR / 'gru' / f'model-part-c-{"_".join([str(i) for i in n_neurons])}'\
                               f'alpha_{alpha}-momentum={momentum}-train_predictions'
    )

    alpha, momentum, n_neurons = part_a_best_params
    part_a_test_preds = load_predictions(
        RNN_LOGS_DIR / 'simple' / f'model-part-a-{"_".join([str(i) for i in n_neurons])}'\
                                  f'alpha_{alpha}-momentum={momentum}-test_predictions'
    )

    alpha, momentum, n_neurons = part_b_best_params
    part_b_test_preds = load_predictions(
        RNN_LOGS_DIR / 'lstm' / f'model-part-b-{"_".join([str(i) for i in n_neurons])}'\
                                f'alpha_{alpha}-momentum={momentum}-test_predictions'
    )

    alpha, momentum, n_neurons = part_c_best_params
    part_c_test_preds = load_predictions(
        RNN_LOGS_DIR / 'gru' / f'model-part-c-{"_".join([str(i) for i in n_neurons])}'\
                               f'alpha_{alpha}-momentum={momentum}-test_predictions'
    )

    index = ['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking']
    histories = [part_a_best_history, part_b_best_history, part_c_best_history]
    all_best_params = [part_a_best_params, part_b_best_params, part_c_best_params]
    all_train_preds = [part_a_train_preds, part_b_train_preds, part_c_train_preds]
    all_test_preds = [part_a_test_preds, part_b_test_preds, part_c_test_preds]
    metrics_set = [['train_cross_entropy', 'valid_cross_entropy'], ['train_accuracy', 'valid_accuracy']]
    names = ['simple', 'lstm', 'gru']

    for i, (name, params, train_preds, test_preds, history) in enumerate(zip(names, all_best_params, all_train_preds, all_test_preds, histories)):
        sns.set(font_scale=1.4)
        alpha, momentum, n_neurons = params
        train_cm = np.zeros((6, 6))
        for pred, true in zip(train_preds, y_train):
            p = pred.argmax()
            t = true.argmax()
            train_cm[p][t] += 1
        train_cm /= len(y_train)
        train_cm = pd.DataFrame(train_cm, index=index, columns=index)

        plt.figure(figsize=(12, 10))
        sns.heatmap(train_cm, annot=True)
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        plt.savefig(RNN_LOGS_DIR / name / f'train_cm-alpha={alpha}-momentum={momentum}-n_neurons={"_".join([str(x) for x in n_neurons])}.png')

        test_cm = np.zeros((6, 6))
        test_acc = 0
        for pred, true in zip(test_preds, y_test):
            p = pred.argmax()
            t = true.argmax()
            test_cm[p][t] += 1
            test_acc += int(p == t)
        test_cm /= len(y_test)
        test_cm = pd.DataFrame(test_cm, index=index, columns=index)
        test_acc /= len(y_test)
        plt.figure(figsize=(12, 10))
        sns.heatmap(test_cm, annot=True)
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        plt.savefig(RNN_LOGS_DIR / name / f'test_cm-test-acc-{test_acc}.png')

        for j, metrics in enumerate(metrics_set):
            plt.figure()
            plt.title(f"{'Cross Entropy' if j == 0 else 'Accuracy'} vs Epochs")
            plt.xlabel('Epochs')
            plt.ylabel(f"{'Cross Entropy' if j == 0 else 'Accuracy'}")
            for metric in metrics:
                history = histories[i]
                plt.plot(history[metric], label='Validation' if 'valid' in metric else 'Training')
            plt.legend()
            plt.savefig(RNN_LOGS_DIR / name / f'{name}-{"cross_entropy" if j == 0 else "accuracy"}.png')