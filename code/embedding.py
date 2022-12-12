# embedding.py
# Solution for Question 2

from collections import defaultdict, namedtuple
from pathlib import Path
import json

from tqdm import tqdm
import h5py
import numpy as np

# IMPORTANT: Please change this according to your own local paths to run the code
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent  # The root directory of the project
DATA_DIR = PROJECT_ROOT_DIR / 'data'  # The directory that contains the data
LOGS_DIR = PROJECT_ROOT_DIR / 'logs'  # The directory that contains the logs
EMBEDDING_DATA_PATH = DATA_DIR / 'data2.h5'  # The filepath to the dataset for q1
EMBEDDING_LOGS_DIR = LOGS_DIR / 'embedding'  # The directory to which the logs will be saved for q1


# Weights and Gradients containers for readability
Weights = namedtuple('Weights', ['W1', 'W2', 'b1', 'b2'])  # the network weights in the order W1, W2, b1, b2
Gradients = Weights                                        # the gradients for the weights in the same order as above


def encode_one_hot(X):
    """One hot encodes the given data"""
    enc = np.identity(X.max() + 1, dtype=np.int32)
    return enc[X]


def load_and_preprocess_data(filepath=EMBEDDING_DATA_PATH):
    with h5py.File(filepath) as file:
        X_train = np.asarray(file.get('trainx'), dtype=np.int32) - 1
        y_train = np.asarray(file.get('traind'), dtype=np.int32) - 1
        X_valid = np.asarray(file.get('valx'), dtype=np.int32) - 1
        y_valid = np.asarray(file.get('vald'), dtype=np.int32) - 1
        X_test = np.asarray(file.get('testx'), dtype=np.int32) - 1
        y_test = np.asarray(file.get('testd'), dtype=np.int32) - 1
        words = np.array([word.decode() for word in file.get('words')])
    X_train = np.transpose([encode_one_hot(X_train[..., i]) for i in range(3)], axes=(1, 2, 0))
    y_train = encode_one_hot(y_train)
    X_valid = np.transpose([encode_one_hot(X_valid[..., i]) for i in range(3)], axes=(1, 2, 0))
    y_valid = encode_one_hot(y_valid)
    X_test = np.transpose([encode_one_hot(X_test[..., i]) for i in range(3)], axes=(1, 2, 0))
    y_test = encode_one_hot(y_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, words


def softmax(z):
    exp = np.exp(z - z.max())
    return exp / exp.sum(axis=0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def categorical_cross_entropy(y_pred, y_true):
    """Calculates the categorical cross entropy loss"""
    masked = np.multiply(y_pred, y_true)
    logloss = np.where(masked != 0, np.log(masked, where=masked != 0), 0)
    return -np.sum(logloss, axis=1)


def accuracy_score(y_pred, y_true):
    labels = y_true.argmax(axis=1)
    preds = y_pred.argmax(axis=1)
    return np.mean(labels == preds)


class EmbeddingNN:
    def __init__(self, D, hidden_units, output_units=250):
        self.D = D
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.R = None
        self.weights = None

    def initialize_weights(self):
        """Initializes the R matrix, the bias and the weights of the network"""
        rng = np.random.default_rng()
        W1 = rng.normal(0, 0.01, size=(self.hidden_units, 3 * self.D))
        W2 = rng.normal(0, 0.01, size=(self.output_units, self.hidden_units))
        b1 = rng.normal(0, 0.01, size=(self.hidden_units, 1))
        b2 = rng.normal(0, 0.01, size=(self.output_units, 1))

        self.R = rng.normal(0, 0.01, (self.D, 250))
        self.weights = Weights(W1, W2, b1, b2)

    def __call__(self, X):
        """Inference mode forward pass through the network"""
        Z = (self.R @ X).reshape(len(X), -1).T
        Z = sigmoid(self.weights.W1 @ Z + self.weights.b1)
        Z = softmax(self.weights.W2 @ Z + self.weights.b2)
        return Z.T

    def step(self, X, y, batch_size):
        """One step of gradient calculation for a mini batch"""
        embedding_output = (self.R @ X).reshape(batch_size, -1).T
        hidden_output = sigmoid(self.weights.W1 @ embedding_output + self.weights.b1)
        network_output = softmax(self.weights.W2 @ hidden_output + self.weights.b2)

        J = np.mean(categorical_cross_entropy(network_output.T, y.T))
        accuracy = accuracy_score(network_output.T, y.T)

        # output layer gradients
        delta = network_output - y
        dW2 = delta @ hidden_output.T / self.output_units
        db2 = np.mean(delta, axis=1, keepdims=True)

        # hidden layer gradients
        delta = (self.weights.W2.T @ delta) * hidden_output * (1 - hidden_output)
        dW1 = delta @ embedding_output.T / self.hidden_units
        db1 = np.mean(delta, axis=1, keepdims=True)

        # embedding gradients
        delta = self.weights.W1.T @ delta
        dR = np.mean([delta[self.D * i: self.D * (i + 1)] @ X[..., i].reshape(X.shape[0], -1) / batch_size
                      for i in range(3)], axis=0)

        return J, accuracy, dR, Gradients(dW1, dW2, db1, db2)

    def fit(self,
            X,
            y,
            X_valid,
            y_valid,
            alpha=0.15,
            momentum=0.85,
            epochs=50,
            batch_size=200,
            tolerance=3,
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
        :param tolerance: the number of epochs without improvement for early stopping
        :param shuffle: whether to shuffle the data each epoch
        :param X_valid: the validation features
        :param y_valid: the validation labels
        :param cold_start: whether to reinitialize the weights before training
        :return:
        """
        if self.R is None or self.weights is None or cold_start:
            self.initialize_weights()

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
                X_batch = X[batch_indices]
                y_batch = y[batch_indices].T

                # forward and backward passes through the network
                J, train_accuracy, dR, gradients = self.step(X_batch, y_batch, batch_size)

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
        """Calculate the weight updates given previous updates and gradients"""
        delta_R = momentum * delta_R_prev + (1 - momentum) * dR if delta_weights_prev else dR
        delta_weights = Weights(*[momentum * delta_b_prev_ + (1 - momentum) * db_
                                  for delta_b_prev_, db_ in zip(delta_weights_prev, gradients)]
                                ) if delta_weights_prev else gradients
        return delta_R, delta_weights

    def apply_updates(self, delta_R, delta_weights, alpha):
        """Apply the calculated updates to the network weights"""
        self.R -= alpha * delta_R
        self.weights = Weights(*[w - alpha * delta for w, delta in zip(self.weights, delta_weights)])


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test, words = load_and_preprocess_data()

    D_P_values = [(8, 64), (16, 128), (32, 256)]
    for D, P in D_P_values:
        model = EmbeddingNN(D, P)
        history = model.fit(X_train, y_train, X_valid, y_valid, alpha=0.15, momentum=0.85)
        preds = model(X_test)

        history_filepath = EMBEDDING_LOGS_DIR / f'model-D={D}-P={P}-history'
        predictions_filepath = EMBEDDING_LOGS_DIR / f'model-D={D}-P={P}-test_predictions'

        with open(history_filepath, 'w') as f:
            json.dump(history, f)

        with open(predictions_filepath, 'wb') as f:
            np.save(f, preds)
