import sys, random
import numpy as np
import reader

"""
Perform one layer of transformer inference on a dataset
using embeddings, positional_embeddings, and weight matrices 
specified in the file model.json
"""


def softmax(logits):
    """
    Return the row-wise softmax of a matrix.
    @param:
    logits - any numpy array
    @return:
    probs - probs[i,j] = exp(logits[i,j])/sum(exp(logits[i,:])), but
      be careful to normalize so that you avoid overflow errors!
    """
    v = np.array(logits)

    # Expand v to 2-d if logits is a 1-d list
    if v.ndim == 1:
        v = np.expand_dims(v, axis=0)

    numerator = np.exp(v - np.max(v, axis=1, keepdims=True))
    probs = numerator / np.sum(numerator, axis=1, keepdims=True)

    return probs


def forward(XK, XQ, WK, WO, WQ, WV):
    """
    Perform one layer of transformer inference, using trained model, on given data.

    @param:
    XK - (T-2)-by-V array containing embeddings of words to be used for keys and values
    XQ - 2-by-V array containing embeddings of words to be used for queries
    WK - V-by-d array mapping X to K
    WO - d-by-V array mapping C to O
    WQ - V-by-d array mapping X to Q
    WV - V-by-d array mapping X to V

    @return:
    A - 2-by-(T-2) array, A[i,j] is attention the i'th query pays to the j'th key
    C - 2-by-d array, context vectors from which P is computed
    K - (T-2)-by-d array, key vectors computed from XK
    O - 2-by-V array, O[i,j] is probability that i'th output word should be j
    Q - 2-by-d array, query vectors computed from XQ
    V - (T-2)-by-d array, value vectors computed from XK
    """
    Q = XQ @ WQ  # 2, d
    K = XK @ WK  # (T-2), d
    A = softmax(Q @ K.T)  # 2, (T-2)
    V = XK @ WV  # (T-2), d
    C = A @ V  # 2, d
    O = softmax(C @ WO)  # 2, v

    return A, C, K, O, Q, V


def generate(embeddings, vocabulary, WK, WO, WQ, WV):
    """
    Perform inference on the provided embeddings, and report the generated sentences.

    @param:
    embeddings - a list of one-hot embedding matrices, one per sentence
    vocabulary - a list of words in the vocabulary
    WK - V-by-d array mapping X to K
    WO - d-by-V array mapping C to O
    WQ - V-by-d array mapping X to Q
    WV - V-by-d array mapping X to V

    @return:
    generated - a list of generated sentences, each as a list of space-separated words.
      The first T-2 words of each sentence should be vocabulary items indexed by the
      argmax of the first T-2 embeddings.  The last 2 words of each sentence should be
      vocabulary items indexed by the argmax of the two outputs computed by running
      the transformer with the provided WK, WO, WQ, and WV.
    """
    generated = []
    for embedding in embeddings:
        XK, XQ, _ = reader.define_task(embedding)
        _, _, _, O, _, _ = forward(XK, XQ, WK, WO, WQ, WV)

        initial_word_indices, inferred_word_indces = np.argmax(XK, axis=1), np.argmax(
            O, axis=1
        )
        words = [vocabulary[idx] for idx in initial_word_indices] + [
            vocabulary[idx] for idx in inferred_word_indces
        ]
        generated.append(words)

    return generated


def cross_entropy_loss(O, Y):
    """
    Calculate losses from network outputs O if target one-hot vectors are Y.

    @param:
    O - NQ-by-V array.  O[n,v]=probability that n'th output is v.
    Y - NQ-by-V array. Y[n,v]=1 if n'th target is v, else Y[n,v]=0.

    @return:
    L - cross-entropy loss, summed over all rows
    dO - NQ-by-V array.  Derivatives of the loss with respect to the elements of O.
    """
    eps = sys.float_info.min
    O_clipped = np.clip(O, eps, None)

    L = -np.sum(Y * np.log(O_clipped))
    dO = -np.divide(Y, O_clipped)

    return L, dO


def gradient(XK, XQ, Y, WK, WO, WQ, WV, A, C, K, O, Q, V):
    """
    Compute gradient of cross-entropy loss with respect to WK, WO, WQ, and WV
    given the input data in K, Q, and V, and the target outputs in Y.

    @param:
    XK - one embedding per row, first n-2 words in the sentence
    XQ - one embedding per row, 3rd-from-last and 2nd-from-last words in the sentence
    Y - one embedding per row, last two words in the sentence
    O - 2-by-V array, O[i,j] is probability that i'th output word should be j
    C - 2-by-d array, context vectors from which O is computed
    V - (T-2)-by-d array, value vectors of which each row of C is a weighted average
    A - 2-by-(T-2) array, A[i,j] is attention the i'th query pays to the j'th key
    K - (T-2)-by-d array, key vectors computed from XK
    Q - 2-by-d array, query vectors computed from XQ

    @return:
    dWK - gradient of cross-entropy with respect to WK
    dWO - gradient of cross-entropy with respect to WO
    dWQ - gradient of cross-entropy with respect to WQ
    dWV - gradient of cross-entropy with respect to WV
    """
    dO = -np.divide(Y, O)
    dS = O * (dO - np.sum(dO * O, axis=-1, keepdims=True))
    dWO = C.T @ dS
    dC = dS @ WO.T

    dV = A.T @ dC
    dWV = XK.T @ dV

    dA = dC @ V.T
    dZ = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
    dQ = dZ @ K
    dK = dZ.T @ Q
    dWQ = XQ.T @ dQ
    dWK = XK.T @ dK

    return dWK, dWO, dWQ, dWV


def train(embeddings, WK, WO, WQ, WV, learningrate, num_iters):
    """
    Train a transformer using stochastic gradient descent (SGD).
    Each iteration of SGD should choose one training sentence, uniformly at random,
    compute the loss and loss gradient for that one sentence,
    then adjust the parameters WK, WO, WQ and WV in the direction of the negative
    gradient scaled by the learningrate.

    @param:
    embeddings - embeddings[i][j,:] is one-hot vector of the j'th word in the i'th training sentence
    WK - the matrix that multiplies each embedding to produce a key
    WO - the matrix that multiplies the context vector to produce an output logit vector
    WQ - the matrix that multiplies each embedding to produce a query
    WV - the matrix that multiplies each embedding to produce a value
    learningrate - scalar learning rate
    num_iters - number of iterations of SGD to perform

    @return:
    losses - losses[t]=cross-entropy loss of t'th iteration
    WK - what WK has become after num_iters of training
    WO - what WO has become after num_iters of training
    WQ - what WQ has become after num_iters of training
    WV - what WV has become after num_iters of training
    """
    losses = []
    for _ in range(num_iters):
        idx = np.random.randint(0, len(embeddings))
        embedding = embeddings[idx]
        XK, XQ, Y = reader.define_task(embedding)
        A, C, K, O, Q, V = forward(XK, XQ, WK, WO, WQ, WV)
        L, _ = cross_entropy_loss(O, Y)
        losses.append(L)
        dWK, dWO, dWQ, dWV = gradient(XK, XQ, Y, WK, WO, WQ, WV, A, C, K, O, Q, V)
        WK = WK - learningrate * dWK
        WO = WO - learningrate * dWO
        WQ = WQ - learningrate * dWQ
        WV = WV - learningrate * dWV

    return losses, WK, WO, WQ, WV
