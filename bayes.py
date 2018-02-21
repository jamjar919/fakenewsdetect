import numpy as np;
from keras.preprocessing.text import text_to_word_sequence;

def probabilityOfWordGivenClass(word, frequencies, vocab, totalFrequencies, totalVocab):
    '''
    We use laplace smoothing here to avoid getting zero probabilities. We also take extra params that only need
    to be calculated once to speed up the function.
    '''
    try:
        index = vocab.index(word);
        freq = frequencies[index];
        prob = (freq + 1)/(totalFrequencies + totalVocab);
        return prob;
    except ValueError:
        return (1)/(totalFrequencies + totalVocab);

def probabilityOfClassGivenDocument(document, c, totalFrequencies, totalVocab, featureExtractor=text_to_word_sequence):
    frequencies, terms, P_c, _ = c;
    # Bayes rule for class c and document d - P(c|d) = P(d|c)P(c)/P(d)
    # We can get our class mapping by representing the document d as a set of features x1,x2...xn
    # Hence calculate P(x1|c)P(x2|c)...P(xn|c)

    print("TOTAL FREQ",totalFrequencies,"TOTAL VOCAB",totalVocab)
    
    # Extract features from our document using whatever feature extractor
    features = featureExtractor(document);
    probabilities = list();
    for f in features:
        probabilities.append(probabilityOfWordGivenClass(f, frequencies, terms, totalFrequencies, totalVocab));
    pi = np.log(1);
    for p in probabilities:
        pi = pi + np.log(p);
    return pi + np.log(P_c);
