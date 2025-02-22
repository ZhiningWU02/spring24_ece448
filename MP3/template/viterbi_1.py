"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5  # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0)  # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))  # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))  # {tag0:{tag1: # }}

    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    # Count occurrences of tags, tag pairs, tag/word pairs
    init_counts = Counter(sentence[0][1] for sentence in sentences)
    emit_counts = Counter(
        (word, tag) for sentence in sentences for word, tag in sentence
    )
    trans_counts = Counter(
        (sentence[i - 1][1], sentence[i][1])
        for sentence in sentences
        for i in range(1, len(sentence))
    )

    tag_counts = Counter(tag for sentence in sentences for _, tag in sentence)
    tag_words = defaultdict(set)
    for sentence in sentences:
        for word, tag in sentence:
            tag_words[tag].add(word)

    # Compute smoothed probabilities
    total_init_counts = sum(init_counts.values())
    all_tags = set(tag for sentence in sentences for _, tag in sentence)
    V_init = len(all_tags)

    init_prob = {
        tag: (init_counts.get(tag, 0) + epsilon_for_pt)
        / (total_init_counts + epsilon_for_pt * (V_init + 1))
        for tag in all_tags
    }

    for tag in tag_counts:
        V_T = len(tag_words[tag])
        n_T = tag_counts[tag]
        emit_prob[tag] = {
            word: (emit_counts[(word, tag)] + emit_epsilon)
            / (n_T + emit_epsilon * (V_T + 1))
            for word in tag_words[tag]
        }
        emit_prob[tag]["UNKNOWN"] = emit_epsilon / (
            n_T + emit_epsilon * (V_T + 1)
        )  # Deal with unknown word

    for tag0 in tag_counts:
        V_trans = len(tag_counts)
        n_tag0 = tag_counts[tag0]
        trans_prob[tag0] = {
            tag1: (trans_counts[(tag0, tag1)] + epsilon_for_pt)
            / (n_tag0 + epsilon_for_pt * (V_trans + 1))
            for tag1 in tag_counts
        }
        trans_prob[tag0]["UNKNOWN"] = epsilon_for_pt / (
            n_tag0 + epsilon_for_pt * (V_trans + 1)
        )

    return init_prob, emit_prob, trans_prob


def viterbi_stepforward(
    i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob
):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = (
        {}
    )  # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = (
        {}
    )  # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    # Calculate the max probability for each tag in column i
    if i == 0:
        for tag in emit_prob.keys():
            emit_p = emit_prob[tag].get(word, emit_prob[tag]["UNKNOWN"])
            log_prob[tag] = prev_prob[tag] + log(emit_p)
            predict_tag_seq[tag] = [tag]
    else:
        for tag in emit_prob.keys():
            max_p = float("-inf")
            best_prev_tag = None

            emit_p = emit_prob[tag].get(word, emit_prob[tag]["UNKNOWN"])
            log_emit_p = log(emit_p)

            for prev_tag in prev_prob:
                log_trans_p = log(
                    trans_prob[prev_tag].get(tag, trans_prob[prev_tag]["UNKNOWN"])
                )

                prob = prev_prob[prev_tag] + log_trans_p + log_emit_p

                if prob > max_p:
                    max_p = prob
                    best_prev_tag = prev_tag

            log_prob[tag] = max_p
            predict_tag_seq[tag] = prev_predict_tag_seq[best_prev_tag] + [tag]

    return log_prob, predict_tag_seq


def viterbi_1(train, test, get_probs=training):
    """
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    init_prob, emit_prob, trans_prob = get_probs(train)

    predicts = []

    for sen in range(len(test)):
        sentence = test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(
                i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob
            )

        # TODO:(III)
        # according to the storage of probabilities and sequences, get the final prediction.
        best_tag = max(log_prob, key=log_prob.get)
        best_tag_seq = predict_tag_seq[best_tag]
        predicts.append(list(zip(sentence, best_tag_seq)))

    return predicts
