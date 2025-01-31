# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter, defaultdict


"""
util for printing values
"""


def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")


"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""


def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(
        trainingdir, testdir, stemming, lowercase, silently
    )
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""


def naive_bayes(
    train_labels, train_data, dev_data, laplace=1.25, pos_prior=0.75, silently=False
):
    print_values(laplace, pos_prior)

    """
    My MP0 code here
    """
    # Process the training data
    label_counts = Counter(train_labels)
    word_counts = defaultdict(lambda: defaultdict(int))
    for i, label in enumerate(tqdm(train_labels, disable=silently)):
        doc_counter = Counter(train_data[i])
        for word, count in doc_counter.items():
            word_counts[label][word] += count

    # Calculating the priors
    priors = {
        label: math.log(label_counts[label] / sum(label_counts.values()))
        for label in tqdm(sorted(label_counts), disable=silently)
    }

    # Calculating the likelihoods
    vocabulary = len(set([word for doc in train_data for word in doc]))

    likelihoods = defaultdict(lambda: defaultdict(float))
    for label in tqdm(sorted(label_counts), disable=silently):
        for word in word_counts[label]:
            likelihoods[label][word] = math.log(
                (word_counts[label][word] + laplace)
                / (sum(word_counts[label].values()) + laplace * vocabulary)
            )

    # Use the pro_prior for development
    priors = {0: math.log(1 - pos_prior), 1: math.log(pos_prior)}

    yhats = []

    for doc in tqdm(dev_data, disable=silently):
        max_probability = {"label": 0, "score": -1e100}
        for label in sorted(label_counts):
            probability = priors[label]
            oov_probability = {
                label: math.log(
                    laplace / (sum(word_counts[label].values()) + laplace * vocabulary)
                )
                for label in sorted(label_counts)
            }

            doc_counter = Counter(doc)
            for word, count in doc_counter.items():
                probability += count * likelihoods[label].get(
                    word, oov_probability[label]
                )  # Handle "out of vocabulary words"

            if probability > max_probability["score"]:
                max_probability = {"label": label, "score": probability}

        yhats.append(max_probability["label"])

    return yhats
