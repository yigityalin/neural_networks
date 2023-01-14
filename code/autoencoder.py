# autoencoder.py
# Solution for Question 1

from collections import namedtuple
from itertools import product
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# IMPORTANT: Please change this according to your own local paths to run the code
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent  # The root directory of the project
DATA_DIR = PROJECT_ROOT_DIR / 'data'                       # The directory that contains the data
LOGS_DIR = PROJECT_ROOT_DIR / 'logs'                       # The directory that contains the logs
AUTOENCODER_DATA_PATH = DATA_DIR / 'data1.h5'              # The filepath to the dataset for q1
AUTOENCODER_LOGS_DIR = LOGS_DIR / 'autoencoder'             # The directory to which the logs will be saved for q1

LOGS_DIR.mkdir(exist_ok=True)
AUTOENCODER_LOGS_DIR.mkdir(exist_ok=True)


# Weights and Gradients containers for readability
Weights = namedtuple('Weights', ['W1', 'W2', 'b1', 'b2'])  # the network weights in the order W1, W2, b1, b2
Gradients = Weights                                        # the gradients for the weights in the same order as above


def load_and_preprocess_images(filepath=AUTOENCODER_DATA_PATH):
    """Preprocesses the images as described in the question"""
    with h5py.File(filepath) as file:
        images = np.asarray(file.get('data'), dtype=np.float64)
    X = 0.2126 * images[:, 0, ...] + 0.7152 * images[:, 1, ...] + 0.0722 * images[:, 2, ...]
    X -= X.mean(axis=(1, 2), keepdims=True)
    X = np.clip(X, -3 * X.std(), 3 * X.std())
    X = 0.8 * (X - np.min(X)) / (np.max(X) - np.min(X)) + 0.1

    images = (images - np.min(images, axis=(1, 2, 3), keepdims=True))\
             / (np.max(images, axis=(1, 2, 3), keepdims=True) - np.min(images, axis=(1, 2, 3), keepdims=True) + 1e-14)
    return images, X


def sigmoid(z):
    """The sigmoid activation function"""
    return 1 / (1 + np.exp(-z))


def aeCost(We, data, params):
    """Calculates the cost and the gradients with respect to the cost function described in the question"""
    weights = Weights(*We)                                              # use Weights class for readability
    hidden_output = sigmoid(weights.W1 @ data + weights.b1)             # hidden layer output after activation function
    network_output = sigmoid(weights.W2 @ hidden_output + weights.b2)   # network output after activation function
    rho = params['rho']                                                 # the level of sparsity
    rho_b = np.mean(hidden_output, axis=1)                              # the average activation of hidden unit
    mse = np.mean(np.square(network_output - data))                     # mse term
    l2_W1 = np.sum(np.square(weights.W1))                               # regularization term for W1
    l2_W2 = np.sum(np.square(weights.W2))                               # regularization term for W2
    kl_divergence = params['beta'] * np.sum(rho * np.log(rho / rho_b) + (1 - rho) * np.log((1 - rho) / (1 - rho_b)))

    J = mse / 2 + params['lambda'] * (l2_W1 + l2_W2) / 2 + params['beta'] * kl_divergence  # the total cost

    delta = (network_output - data) * network_output * (1 - network_output)                # dLoss / dZ
    dW2 = delta @ hidden_output.T / params['Lin'] + params['lambda'] * weights.W2
    db2 = np.mean(delta, axis=1, keepdims=True)

    kl_divergence_derivative = np.expand_dims((1 - rho) / (1 - rho_b) - rho / rho_b, axis=1)
    delta = (weights.W2.T @ delta + params['beta'] * kl_divergence_derivative) * hidden_output * (1 - hidden_output)
    dW1 = delta @ data.T / params['Lhid'] + params['lambda'] * weights.W1
    db1 = np.mean(delta, axis=1, keepdims=True)
    return J, Gradients(dW1, dW2, db1, db2)


class Autoencoder:
    def __init__(self,
                 hidden_units,
                 lambda_,
                 beta,
                 rho):
        self.visible_units = None           # the number of input & output units (will be determined while training)
        self.hidden_units = hidden_units    # the number of neurons in the hidden layer
        self.lambda_ = lambda_              # the coefficient of the regularization term of cost
        self.beta = beta                    # the coefficient the KL divergence term of the cost
        self.rho = rho                      # the level of sparsity
        self.weights = None                 # the layer weights

    @property
    def params(self):
        """A dictionary containing the hyperparameters of the autoencoder"""
        return {
            'Lin': self.visible_units,
            'Lhid': self.hidden_units,
            'lambda': self.lambda_,
            'beta': self.beta,
            'rho': self.rho
        }

    def initialize_weights(self):
        """Initializes the weights from a uniform distribution from ()"""
        bound = np.sqrt(6 / (self.visible_units + self.hidden_units))  # the bounds of the initialization distribution
        rng = np.random.default_rng()
        W1 = rng.uniform(-bound, bound, size=(self.hidden_units, self.visible_units))
        W2 = rng.uniform(-bound, bound, size=(self.visible_units, self.hidden_units))
        b1 = rng.uniform(-bound, bound, size=(self.hidden_units, 1))
        b2 = rng.uniform(-bound, bound, size=(self.visible_units, 1))
        self.weights = Weights(W1, W2, b1, b2)

    def fit(self,
            X,
            alpha,
            epochs=1,
            batch_size=32,
            shuffle=True,
            cold_start=False
            ):
        """
        Train the neural network
        :param X: the training data
        :param alpha: the learning rate
        :param epochs: the number of training epochs
        :param batch_size: the size of the training batches
        :param shuffle: whether to shuffle the data each epoch
        :param cold_start: whether to reinitialize the weights before training
        :return:
        """
        if X.ndim == 3:
            X = X.reshape(len(X), -1)
        elif X.ndim != 2:
            raise ValueError('X.ndim must be 2 or 3.')

        if self.visible_units is None or self.weights is None or cold_start:
            self.visible_units = X.shape[1]
            self.initialize_weights()

        n_batches = len(X) // batch_size
        for epoch in range(epochs):
            indices = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
            for batch in range(n_batches):
                batch_indices = indices[batch * batch_size: (batch + 1) * batch_size]
                X_batch = X[batch_indices].T
                _, gradients = aeCost(self.weights, X_batch, self.params)
                self.apply_gradients(gradients, alpha)
        return self

    def apply_gradients(self, gradients, alpha):
        """Update the weights of the network"""
        self.weights = Weights(*[w - alpha * delta for w, delta in zip(self.weights, gradients)])

    def plot_hidden_activations(self, n_cols=8, path=None):
        W = self.weights.W1.copy()  # hidden layer weights
        n_features = self.hidden_units
        n_rows = int(np.floor(n_features / n_cols))

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        figshape = int(np.sqrt(self.visible_units))
        for index, w in enumerate(W):
            w = (w - w.min()) / (w.max() - w.min())
            row, col = np.unravel_index(index, ax.shape)
            ax[row, col].set_axis_off()
            ax[row, col].imshow(w.reshape(figshape, figshape).T, cmap='gray')
        if path is not None:
            plt.savefig(path)
        else:
            plt.show()


def display_images(rgb, gray, n, n_cols, save_rgb_filepath, save_gray_filepath):
    rng = np.random.default_rng()
    indices = rng.choice(np.arange(0, len(images)), size=n, replace=False)

    n_rows = int(np.ceil(n // n_cols))
    rgb_fig, rgb_ax = plt.subplots(n_rows, n_cols)
    gray_fig, gray_ax = plt.subplots(n_rows, n_cols)

    rgb_images = rgb[indices]
    gray_images = gray[indices]
    for index, (rgb_im, gray_im) in enumerate(zip(rgb_images, gray_images)):
        row, col = np.unravel_index(index, rgb_ax.shape)
        rgb_ax[row, col].set_axis_off()
        rgb_ax[row, col].imshow(rgb_im.T)
        gray_ax[row, col].set_axis_off()
        gray_ax[row, col].imshow(gray_im.T, cmap='gray',)
    rgb_fig.savefig(save_rgb_filepath)
    gray_fig.savefig(save_gray_filepath)


def q1_main_part_a(images, X):
    rgb_filepath = AUTOENCODER_LOGS_DIR / 'rgb_images.png'
    gray_filepath = AUTOENCODER_LOGS_DIR / 'gray_images.png'

    display_images(images, X, n=200, n_cols=8, save_rgb_filepath=rgb_filepath, save_gray_filepath=gray_filepath)


def q1_main_part_b(X):
    betas = [0.05, 0.1, 0.2]
    rhos = [0.05, 0.1, 0.5]
    for beta, rho in product(betas, rhos):
        model = Autoencoder(hidden_units=64, lambda_=5e-4, beta=beta, rho=rho)
        model.fit(X, alpha=0.05, epochs=500)
        model.fit(X, alpha=0.01, epochs=100)
        path = AUTOENCODER_LOGS_DIR / f'hidden_activations-beta_{beta:.0e}-rho_{rho:.0e}.png'
        model.plot_hidden_activations(path=path)


def q1_main_part_d(X):
    beta = 0.2
    rho = 0.05
    Lhids = [32, 64, 96]
    lambdas = [5e-3, 1e-4, 5e-4]
    for Lhid, lambda_ in product(Lhids, lambdas):
        model = Autoencoder(hidden_units=Lhid, lambda_=lambda_, beta=beta, rho=rho)
        model.fit(X, alpha=0.05, epochs=500)
        model.fit(X, alpha=0.01, epochs=100)
        path = AUTOENCODER_LOGS_DIR / f'hidden_activations-Lhid_{Lhid}-lambda_{lambda_:.0e}' \
                                      f'-beta_{beta:.0e}-rho_{rho:.0e}.png'
        model.plot_hidden_activations(path=path)


if __name__ == '__main__':
    images, X = load_and_preprocess_images()
    q1_main_part_a(images, X)
    q1_main_part_b(X)
    q1_main_part_d(X)
