import numpy as np
from keras.preprocessing.text import text_to_word_sequence

def probabilityOfWordGivenClass(word, frequencies, vocab, totalFrequencies, totalVocab):
    '''
    We use laplace smoothing here to avoid getting zero probabilities. We also take extra params that only need
    to be calculated once to speed up the function.
    '''
    try:
        index = vocab.index(word)
        freq = frequencies[index]
        prob = (freq + 1)/(totalFrequencies + totalVocab)
        return prob
    except ValueError:
        return (1)/(totalFrequencies + totalVocab)

def probabilityOfClassGivenDocument(document, c, totalFrequencies, totalVocab, ngramSizes=[], featureExtractor=text_to_word_sequence, USE_NGRAMS=False, NOISY=False):
    frequencies, terms, P_c, _, ngrams = c
    # Bayes rule for class c and document d - P(c|d) = P(d|c)P(c)/P(d)
    # We can get our class mapping by representing the document d as a set of features x1,x2...xn
    # Hence calculate P(x1|c)P(x2|c)...P(xn|c)
    
    # Extract features from our document using whatever feature extractor
    features = featureExtractor(document)

    if USE_NGRAMS:
        # Unpack frquencies and terms
        frequencies2Gram, terms2Gram = ngrams[0]
        frequencies3Gram, terms3Gram = ngrams[1]

        totalFreq2Gram, totalVocab2Gram = ngramSizes[0]
        totalFreq3Gram, totalVocab3Gram = ngramSizes[1]

    probabilities = list()
    for i in range(0, len(features)):
        f = features[i]
        p = probabilityOfWordGivenClass(f, frequencies, terms, totalFrequencies, totalVocab)

        if USE_NGRAMS and i > 2:
            # use formula
            # P(w_n | w_n-1 w_n-2) = 
            # 0.125 P(w_n) + 0.375 P(w_n | w_n-1) + 0.5 P(w_n | w_n-1 w_n-2)
            # Because estimating lamdas is hard
            wm1 = features[i-1]
            wm2 = features[i-2]

            gram2 = wm1 + " " + f
            gram3 = wm2 + " " + gram2

            pgram2 = probabilityOfWordGivenClass(gram2, frequencies2Gram, terms2Gram, totalFreq2Gram, totalVocab2Gram)
            pgram3 = probabilityOfWordGivenClass(gram3, frequencies3Gram, terms3Gram, totalFreq3Gram, totalVocab3Gram)
            
            p = 0.125*p + 0.375*pgram2 + 0.5*pgram3

        probabilities.append(p)
    pi = np.log(1)
    for p in probabilities:
        pi = pi + np.log(p)
    return pi + np.log(P_c)
