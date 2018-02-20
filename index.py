import pandas as pd 
from keras.preprocessing.text import text_to_word_sequence
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from functools import reduce


# Constants
NUMBER_OF_TEST_ARTICLES = 100;
VOCAB_SIZE = 1000;

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
    print("TOTAL FREQ",totalFrequencies,"TOTAL VOCAB",totalVocab)
    # Find word in frequencies list
    try:
        print("Looking for frequency of",word, end=" ")
        index = vocab.index(word);
        freq = frequencies[index];
        print("Frequency is",freq, end=" ")
        prob = (freq + 1)/(totalFrequencies + totalVocab);
        print("Prob is ",prob)
        return prob;
    except ValueError:
        print("DID NOT FIND IN VOCAB")
        return 1;

def probabilityOfClassGivenDocument(document, c, totalFrequencies, totalVocab):
    frequencies, terms, P_c, _ = c;
    # Bayes rule for class c and document d - P(c|d) = P(d|c)P(c)/P(d)
    # We can get our class mapping by representing the document d as a set of features x1,x2...xn
    # Hence calculate P(x1|c)P(x2|c)...P(xn|c)

    print(totalVocab)
    
    # Extract words from our document
    features = text_to_word_sequence(document);
    probabilities = list();
    for f in features:
        probabilities.append(probabilityOfWordGivenClass(f, frequencies, terms, totalFrequencies, totalVocab));
    pi = np.float64(1);
    for p in probabilities:
        pi = pi * np.float64(p);
    return pi*np.float64(P_c);

def documentClass(document, classes = []):
    # Classes = An array of (frequencies, terms, P_c, name)
    # Calculate vocab sizes and total the frequencies for each class
    vocabSizes = list();
    totalFrequencies = list();
    for c in classes:
        vocabSizes.append(len(c[1]));
        print(len(c[1]))
        print(len(c[0]))
        totalFrequencies.append(np.sum(c[0]));
    # Match the document to each class:
    print("Testing document");
    print(document[0:300]);
    for i in range(0, len(classes)):
        c = classes[i];
        prob = probabilityOfClassGivenDocument(cleanText(document), c, totalFrequencies[i], vocabSizes[i]);
        print("Document has a "+str(prob)+" chance of being in class "+c[3]);

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
print("vocab")
print(terms)

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
numCorrectPositive = 0;
numFalsePositive = 0;
numFalseNegative = 0;
numCorrectNegative = 0;

# Test with first 5 fake documents and first 5 real
for i in range(0, 3):#test["TEXT"].size):
    # Test real one 
    document = test["TEXT"].iloc[i];
    c = documentClass(document, [(frequencySumFake, termsFake, P_fake, "Fake"), (frequencySumReal, termsReal, P_real, "Real")])
    
