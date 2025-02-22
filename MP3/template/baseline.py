"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""


def baseline(train, test):
    """
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    priors = {}
    likelihoods = {}

    for word, tag in (pair for sentence in train for pair in sentence):
        if word not in priors:
            priors[word] = {}
        priors[word][tag] = priors[word].get(tag, 0) + 1
        likelihoods[tag] = likelihoods.get(tag, 0) + 1

    # Calculate the likelihoods and priors
    def predict(w):
        if w in priors:
            word_tag_counts = priors[w]
            return max(word_tag_counts, key=word_tag_counts.get)
        else:
            return max(likelihoods, key=likelihoods.get)

    # Infer the tags of the test samples
    prediction = []
    for sentence in test:
        pred_sentence = [(word, predict(word)) for word in sentence]
        prediction.append(pred_sentence)

    return prediction
