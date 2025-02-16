import numpy as np


def initialize(data, dim):
    """
    Initialize embeddings for all distinct words in the input data.
    Most of the dimensions will be zero-mean unit-variance Gaussian random variables.
    In order to make debugging easier, however, we will assign special geometric values
    to the first two dimensions of the embedding:

    (1) Find out how many distinct words there are.
    (2) Choose that many locations uniformly spaced on a unit circle in the first two dimensions.
    (3) Put the words into those spots in the same order that they occur in the data.

    Thus if data[0] and data[1] are different words, you should have

    embedding[data[0]] = np.array([np.cos(0), np.sin(0), random, random, random, ...])
    embedding[data[1]] = np.array([np.cos(2*np.pi/N), np.sin(2*np.pi/N), random, random, random, ...])

    ... and so on, where N is the number of distinct words, and each random element is
    a Gaussian random variable with mean=0 and standard deviation=1.

    @param:
    data (list) - list of words in the input text, split on whitespace
    dim (int) - dimension of the learned embeddings

    @return:
    embedding - dict mapping from words (strings) to numpy arrays of dimension=dim.
    """
    words = list(set(data))
    n_data = len(words)
    if n_data == 0:
        return {}  # Return empty dict if there is no word to embed

    embedding = {}
    for i, word in enumerate(words):
        arr = np.array([np.cos(i * 2 * np.pi / n_data), np.sin(i * 2 * np.pi / n_data)])
        embedding[word] = np.hstack((arr, np.random.normal(0, 1, dim - 2)))

    return embedding


def gradient(embedding, data, t, d, k):
    """
    Calculate gradient of the skipgram NCE loss with respect to the embedding of data[t]

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    t (int) - data index of word with respect to which you want the gradient
    d (int) - choose context words from t-d through t+d, not including t
    k (int) - compare each context word to k words chosen uniformly at random from the data

    @return:
    g (numpy array) - loss gradients with respect to embedding of data[t]
    """
    g = np.zeros(embedding[data[0]].shape)
    for j in range(-d, d + 1):
        if j == 0 or t + j < 0 or t + j >= len(data):
            continue  # Continue if index out of range or data[t + j] equals data [t]

        g -= (
            1 - 1 / (1 + np.exp(-embedding[data[t + j]].T @ embedding[data[t]]))
        ) * embedding[data[t + j]]

        noise = np.random.randint(0, len(data), k)
        for i in noise:
            g += (
                1
                / k
                * 1
                / (1 + np.exp(-embedding[data[i]].T @ embedding[data[t]]))
                * embedding[data[i]]
            )

    return g


def sgd(embedding, data, learning_rate, num_iters, d, k):
    """
    Perform num_iters steps of stochastic gradient descent.

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    learning_rate (scalar) - scale the negative gradient by this amount at each step
    num_iters (int) - the number of iterations to perform
    d (int) - context width hyperparameter for gradient computation
    k (int) - noise sample size hyperparameter for gradient computation

    @return:
    embedding - the updated embeddings
    """
    for i in np.random.randint(0, len(data), num_iters):
        g = gradient(embedding, data, i, d, k)
        embedding[data[i]] -= learning_rate * g

    return embedding
