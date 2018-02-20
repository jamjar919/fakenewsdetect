import pandas as pd 
from keras.preprocessing.text import text_to_word_sequence
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
#nltk.download();

def loadArticles(name)
    return pd.read_csv(name, header=0, delimiter=",")

def cleanText(dirty):
    # Convert to lowercase and remove punctuation
    clean = text_to_word_sequence(dirty);
    # Remove stopwords
    stops = set(stopwords.words("english"));
    clean = [w for w in clean if not w in stops]
    return (" ".join(clean));

def calculateTermFrequency(corpus, max_features=None, ngrams=1):
    '''
    Returns a list of terms and their count given a list of strings
    '''
    vectorizer = CountVectorizer(
        analyzer = "word",
        tokenizer = None,
        preprocessor = None,
        stop_words = None,
        max_features = max_features,
        ngram_range=(ngrams,ngrams)
    );
    features = vectorizer.fit_transform(corpus);
    return (features.toarray(), vectorizer.get_feature_names());

def printFrequencyCount(frequency, terms):
    dist = np.sum(frequency, axis=0)
    for tag, count in zip(terms, dist):
        print(count, tag)

def probabilityOfWordGivenClass(word, frequencies, totalFrequencies, totalVocab):
    '''
    We use laplace smoothing here
    '''
    # Find word in frequencies list


articles = loadArticles("news_ds.csv")

# Calculate base probabilities for P_fake, P_real (the chance of an article being classified into these categories)
numArticles = articles["TEXT"].size

fakeArticles = list();
realArticles = list();

# Tokenize the document - this converts to lowercase and removes punctuation. Do this for all fake and all real articles
for i in range(0, numArticles):
    if articles["LABEL"][i] == 0:
        fakeArticles.append(cleanText(articles["TEXT"][i]));
    else:
        realArticles.append(cleanText(articles["TEXT"][i]));

numReal = len(realArticles);
numFake = len(fakeArticles);

print("Tokenised",numReal,"real articles,",numFake,"fake",numArticles,"total");

# Calculate P(c)'s
P_fake = numFake / numArticles;
P_real = numReal / numArticles;

# Calculate ngrams
# Unigrams
frequencyReal, termsReal = calculateTermFrequency(realArticles);
frequencyFake, termsFake = calculateTermFrequency(fakeArticles);

# Vocab sizes
vocabReal = len(termsReal);
vocabFake = len(termsFake);

# Total count of the frequencies
totalFrequenciesReal = np.sum(np.sum(frequencyReal, axis=0));
totalFrequenciesFake = np.sum(np.sum(frequencyFake, axis=0));

#printFrequencyCount(frequency, terms);

# Bayes rule for class c and document d - P(c|d) = P(d|c)P(c)/P(d)
# We can get our class mapping by representing the document d as a set of features x1,x2...xn
# And finding the class c s.t P(x1,x2...xn|c)P(c) is maximised
# Hence calculate P(x1|c)P(x2|c)...P(xn|c) for real and fake classes for our documents


