import pandas as pd 
from keras.preprocessing.text import text_to_word_sequence
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from functools import reduce


# Constants
NUMBER_OF_TEST_ARTICLES = 100;
VOCAB_SIZE = 5000;

def loadArticles(name):
    return pd.read_csv(name, header=0, delimiter=",")

def cleanText(dirty):
    # Convert to lowercase and remove punctuation
    clean = text_to_word_sequence(dirty);
    # Remove stopwords
    stops = set(stopwords.words("english"));
    clean = [w for w in clean if not w in stops]
    return (" ".join(clean));

def calculateTermFrequency(corpus, max_features=None, ngrams=1, vocabulary=None):
    '''
    Returns a list of terms and their count given a list of strings
    '''
    vectorizer = CountVectorizer(
        analyzer = "word",
        tokenizer = None,
        preprocessor = None,
        stop_words = None,
        max_features = max_features,
        ngram_range=(ngrams,ngrams),
        vocabulary=vocabulary
    );
    features = vectorizer.fit_transform(corpus);
    return (features.toarray(), vectorizer.get_feature_names());

def printFrequencyCount(frequency, terms):
    dist = np.sum(frequency, axis=0)
    for tag, count in zip(terms, dist):
        print(count, tag)

def probabilityOfWordGivenClass(word, frequencies, vocab, totalFrequencies, totalVocab):
    '''
    We use laplace smoothing here
    '''
    # Find word in frequencies list
    try:
        index = vocab.index(word);
        freq = frequencies[index];
        prob = (freq + 1)/(totalFrequencies + totalVocab);
        return prob;
    except ValueError:
        return (1)/(totalFrequencies + totalVocab);

def probabilityOfClassGivenDocument(document, c, totalFrequencies, totalVocab):
    frequencies, terms, P_c, _ = c;
    # Bayes rule for class c and document d - P(c|d) = P(d|c)P(c)/P(d)
    # We can get our class mapping by representing the document d as a set of features x1,x2...xn
    # Hence calculate P(x1|c)P(x2|c)...P(xn|c)

    print("TOTAL FREQ",totalFrequencies,"TOTAL VOCAB",totalVocab)
    
    # Extract words from our document
    features = text_to_word_sequence(document);
    probabilities = list();
    for f in features:
        probabilities.append(probabilityOfWordGivenClass(f, frequencies, terms, totalFrequencies, totalVocab));
    pi = np.log(1);
    for p in probabilities:
        pi = pi + np.log(p);
    return pi + np.log(P_c);

def documentClass(document, classes = []):
    # Classes = An array of (frequencies, terms, P_c, name)
    # Calculate vocab sizes and total the frequencies for each class
    # Returns the log of the score
    vocabSizes = list();
    totalFrequencies = list();
    for c in classes:
        vocabSizes.append(len(c[1]));
        totalFrequencies.append(np.sum(c[0]));
    # Match the document to each class:
    print("Testing document");
    print(document[0:300]);
    vals = list();
    for i in range(0, len(classes)):
        c = classes[i];
        prob = probabilityOfClassGivenDocument(cleanText(document), c, totalFrequencies[i], vocabSizes[i]);
        vals.append(prob);
        print("Document has a e^"+str(prob)+" score for being in class "+c[3]+"("+str(i)+")");
    # Find maximum out of vals
    m = np.amax(vals);
    indexOfMax = vals.index(m);
    return indexOfMax;

articles = loadArticles("news_ds.csv")

# Split into training articles and test articles
test = articles[0:NUMBER_OF_TEST_ARTICLES];
articles = articles[NUMBER_OF_TEST_ARTICLES:];

print(test);
print(articles);

# Calculate base probabilities for P_fake, P_real (the chance of an article being classified into these categories)
numArticles = articles["TEXT"].size

fakeArticles = list();
realArticles = list();
allArticles = list();

# Tokenize the document - this converts to lowercase and removes punctuation. Do this for all fake and all real articles
for i in range(0, numArticles):
    clean = cleanText(articles["TEXT"].iloc[i]);
    allArticles.append(clean);
    if articles["LABEL"].iloc[i] == 0:
        fakeArticles.append(clean);
    else:
        realArticles.append(clean);

numReal = len(realArticles);
numFake = len(fakeArticles);

print("Tokenised",numReal,"real articles,",numFake,"fake",numArticles,"total");

print("Extracting vocabulary")
frequency, terms = calculateTermFrequency(allArticles, max_features=VOCAB_SIZE)

# Calculate P(c)'s
P_fake = numFake / numArticles;
P_real = numReal / numArticles;

print("P_fake:",P_fake,"  P_real:",P_real);

# Calculate ngrams
# Unigrams
frequencyReal, termsReal = calculateTermFrequency(realArticles, max_features=VOCAB_SIZE, vocabulary=terms);
frequencyFake, termsFake = calculateTermFrequency(fakeArticles, max_features=VOCAB_SIZE, vocabulary=terms);

# Calculate total frequencies
frequencySumReal = np.sum(frequencyReal, axis=0)
frequencySumFake = np.sum(frequencyFake, axis=0)

# Set up evaluation vars
numCorrectPositive = 0; # We classified as real, and it's real
numFalsePositive = 0; # We classified as real, and it's fake 
numFalseNegative = 0; # We classified as fake, and it's real
numCorrectNegative = 0; # We classified as fake, and it's fake

# Test!!
for i in range(0, test["TEXT"].size):
    # Test real one 
    document = test["TEXT"].iloc[i];
    realClass = test["LABEL"].iloc[i];
    print()
    print()
    print("This document is in class " + ("Fake" if realClass == 0 else "Real" ))
    c = documentClass(document, [(frequencySumFake, termsFake, P_fake, "Fake"), (frequencySumReal, termsReal, P_real, "Real")])
    print("Document was classified as "+ ("Fake" if c == 0 else "Real" ))
    if c == 0:
        if realClass == 0:
            numCorrectNegative += 1;
        if realClass == 0: 
            numFalseNegative += 1;
    if c == 1:
        if realClass == 1:
            numCorrectPositive += 1;
        if realClass == 0: 
            numFalsePositive += 1;

# Work out percents
numCorrectPositivePercent = 100 * (numCorrectPositive / test["TEXT"].size);
numFalsePositivePercent = 100 * (numFalsePositive / test["TEXT"].size);
numFalseNegativePercent = 100 * (numFalseNegative / test["TEXT"].size);
numCorrectNegativePercent = 100 * (numCorrectNegative / test["TEXT"].size);
totalCorrect = numCorrectPositive + numCorrectNegative;
totalCorrectPercent = 100 * (totalCorrectPercent / test["TEXT"].size);

print()
print("Test Run Complete")
print()
print("Total correct: ")


