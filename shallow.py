import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from functools import reduce
from bayes import probabilityOfWordGivenClass, probabilityOfClassGivenDocument
from unidecode import unidecode


def cleanText(dirty):
    # Convert to lowercase and remove punctuation
    dirty = unidecode(str(dirty))
    dirty = dirty.replace("\"", " ")

    clean = text_to_word_sequence(dirty)
    # Remove stopwords
    stops = set(stopwords.words("english"))
    clean = [w for w in clean if not w in stops]
    return (" ".join(clean))

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
    )
    features = vectorizer.fit_transform(corpus)
    return (features.toarray(), vectorizer.get_feature_names())

def printFrequencyCount(frequency, terms):
    dist = np.sum(frequency, axis=0)
    for tag, count in zip(terms, dist):
        print(count, tag)

def documentClass(document, classes = [], USE_NGRAMS=False, NOISY=False):
    '''
    Classes = An tuple of (frequencies, terms, P_c, name*, ngrams*)
    Returns the log of the score, not the actual score (As it's very small)
    '''

    # Calculate vocab sizes and total the frequencies for each class
    vocabSizes = list()
    totalFrequencies = list()
    ngramSizes = list()
    for c in classes:
        vocabSizes.append(len(c[1]))
        totalFrequencies.append(np.sum(c[0]))
        if USE_NGRAMS:
            for ngramData in c[4]:
                f, t = ngramData
                ngramSizes.append((np.sum(f), len(t)))
        if NOISY:
            print("Testing document")
            print(document[0:300],"...")

    # Match the document to each class using bayes.py
    vals = list()
    for i in range(0, len(classes)):
        c = classes[i]
        prob = probabilityOfClassGivenDocument(cleanText(document), c, totalFrequencies[i], vocabSizes[i], ngramSizes=ngramSizes, USE_NGRAMS=USE_NGRAMS, NOISY=NOISY)
        vals.append(prob)
        if NOISY:
            print("Document has a e^"+str(prob)+" score for being in class "+c[3]+"("+str(i)+")")
        
    # Find maximum out of vals
    m = np.amax(vals)
    indexOfMax = vals.index(m)
    return indexOfMax

def shallowLearn(articles, NUMBER_OF_TEST_ARTICLES=500, VOCAB_SIZE=10000, USE_TDIDF=True, USE_NGRAMS=True, NOISY=False):
    # Split into training articles and test articles
    test = articles[0:NUMBER_OF_TEST_ARTICLES]
    articles = articles[NUMBER_OF_TEST_ARTICLES:]

    # Calculate base probabilities for P_fake, P_real (the chance of an article being classified into these categories)
    numArticles = articles["TEXT"].size

    fakeArticles = list()
    realArticles = list()
    allArticles = list()

    # Tokenize the document - this converts to lowercase and removes punctuation. Do this for all fake and all real articles
    for i in range(0, numArticles):
        clean = cleanText(articles["TEXT"].iloc[i])
        allArticles.append(clean)
        if articles["LABEL"].iloc[i] == 0:
            fakeArticles.append(clean)
        else:
            realArticles.append(clean)

    numReal = len(realArticles)
    numFake = len(fakeArticles)

    print("Tokenised",numReal,"real articles,",numFake,"fake",numArticles,"total")

    print("Extracting vocabulary")
    _, terms = calculateTermFrequency(allArticles, max_features=VOCAB_SIZE)
    print("Vocab size",len(terms))
    if (USE_NGRAMS):
        print("Extracting 2-grams")
        _, ngram2Terms = calculateTermFrequency(allArticles, max_features=VOCAB_SIZE, ngrams=2)
        print("Vocab size",len(ngram2Terms))
        print("Extracting 3-grams")
        _, ngram3Terms = calculateTermFrequency(allArticles, max_features=VOCAB_SIZE, ngrams=3)
        print("Vocab size",len(ngram3Terms))
    print("Terms extracted")


    # Calculate P(c)'s
    P_fake = numFake / numArticles
    P_real = numReal / numArticles

    print("P_fake:",P_fake,"  P_real:",P_real)

    # Calculate ngrams
    print("Extracting frequency for fake and real")
    frequencyReal, termsReal = calculateTermFrequency(realArticles, max_features=VOCAB_SIZE, vocabulary=terms)
    frequencyFake, termsFake = calculateTermFrequency(fakeArticles, max_features=VOCAB_SIZE, vocabulary=terms)
    if (USE_NGRAMS):
        print("Extracting 2-grams freq")
        ngram2frequencyReal, ngram2TermsReal = calculateTermFrequency(realArticles, max_features=VOCAB_SIZE, ngrams=2, vocabulary=ngram2Terms)
        ngram2frequencyFake, ngram2TermsFake = calculateTermFrequency(fakeArticles, max_features=VOCAB_SIZE, ngrams=2, vocabulary=ngram2Terms)
        print("Extracting 3-grams freq")
        ngram3frequencyReal, ngram3TermsReal = calculateTermFrequency(realArticles, max_features=VOCAB_SIZE, ngrams=3, vocabulary=ngram3Terms)
        ngram3frequencyFake, ngram3TermsFake = calculateTermFrequency(fakeArticles, max_features=VOCAB_SIZE, ngrams=3, vocabulary=ngram3Terms)
    print("Frequencies extracted")

    # Are we using TD-IDF? 
    if USE_TDIDF:
        print("Convert to TFIDF format")
        # Convert our term frequency into term frequency inverse document frequency
        transformer = TfidfTransformer(norm="l2")
        frequencyReal = transformer.fit_transform(frequencyReal).todense()
        frequencyFake = transformer.fit_transform(frequencyFake).todense()
        if USE_NGRAMS:
            ngram2frequencyReal = transformer.fit_transform(ngram2frequencyReal).todense()
            ngram2frequencyFake = transformer.fit_transform(ngram2frequencyFake).todense()
            ngram3frequencyReal = transformer.fit_transform(ngram3frequencyReal).todense()
            ngram3frequencyFake = transformer.fit_transform(ngram3frequencyFake).todense()

        print("Done")

    # Calculate total frequencies
    frequencySumReal = np.sum(frequencyReal, axis=0)
    frequencySumFake = np.sum(frequencyFake, axis=0)
    if USE_NGRAMS:
        ngram2frequencySumReal = np.sum(ngram2frequencyReal, axis=0)
        ngram2frequencySumFake = np.sum(ngram2frequencyFake, axis=0)
        ngram3frequencySumReal = np.sum(ngram3frequencyReal, axis=0)
        ngram3frequencySumFake = np.sum(ngram3frequencyFake, axis=0)

    # Requires an extra unpack for some reason
    if USE_TDIDF:
        frequencySumReal = frequencySumReal.tolist()[0]
        frequencySumFake = frequencySumFake.tolist()[0]
        if USE_NGRAMS:
            ngram2frequencySumReal = ngram2frequencySumReal.tolist()[0]
            ngram2frequencySumFake = ngram2frequencySumFake.tolist()[0]
            ngram3frequencySumReal = ngram3frequencySumReal.tolist()[0]
            ngram3frequencySumFake = ngram3frequencySumFake.tolist()[0]

    # Set up ngram list if we need to
    ngramListReal = []
    ngramListFake = []
    if USE_NGRAMS:
        ngramListReal = [
            (ngram2frequencySumReal, ngram2TermsReal),
            (ngram3frequencySumReal, ngram3TermsReal)
        ]
        ngramListFake = [
            (ngram2frequencySumFake, ngram2TermsFake),
            (ngram3frequencySumFake, ngram3TermsFake)
        ]

    print("Trained!")

    # Set up evaluation vars
    numCorrectPositive = 0 # We classified as real, and it's real
    numFalsePositive = 0 # We classified as real, and it's fake 
    numFalseNegative = 0 # We classified as fake, and it's real
    numCorrectNegative = 0 # We classified as fake, and it's fake

    # Test!!
    print("Testing data...")
    for i in range(0, test["TEXT"].size):
        # Test real one 
        document = test["TEXT"].iloc[i]
        realClass = test["LABEL"].iloc[i]
        if NOISY:
            print()
            print()
            print("This document is in class " + ("Fake" if realClass == 0 else "Real" ))
        c = documentClass(
            document,
            [
                (frequencySumFake, termsFake, P_fake, "Fake", ngramListFake),
                (frequencySumReal, termsReal, P_real, "Real", ngramListReal)
            ],
            USE_NGRAMS=USE_NGRAMS,
            NOISY=NOISY
        )
        if NOISY:
            print("Document was classified as "+ ("Fake" if c == 0 else "Real" ))
        else:
            stats = " "+str(numCorrectNegative + numCorrectPositive)+" correct, "+str(numFalseNegative + numFalsePositive)+" incorrect"
            if realClass == c:
                print("\r./"+stats, end="")
            else:
                print("\rx "+stats, end="")
        if c == 0:
            if realClass == 0:
                numCorrectNegative += 1
            if realClass == 1: 
                numFalseNegative += 1
        if c == 1:
            if realClass == 1:
                numCorrectPositive += 1
            if realClass == 0: 
                numFalsePositive += 1

    # Work out percents
    numCorrectPositivePercent = 100 * (numCorrectPositive / test["TEXT"].size)
    numFalsePositivePercent = 100 * (numFalsePositive / test["TEXT"].size)
    numFalseNegativePercent = 100 * (numFalseNegative / test["TEXT"].size)
    numCorrectNegativePercent = 100 * (numCorrectNegative / test["TEXT"].size)
    totalCorrect = numCorrectPositive + numCorrectNegative
    totalCorrectPercent = 100 * (totalCorrect / test["TEXT"].size)
    recall = numCorrectPositive / (numCorrectPositive + numFalseNegative)
    precision = numCorrectPositive / (numCorrectPositive + numFalsePositive)
    fmeasure = 2*(recall * precision) / (recall + precision)

    print()
    print("Test Run Complete")
    print()
    print("Total correct: "+str(totalCorrect)+"("+str(totalCorrectPercent)+"%)")
    print("Total correct positives: "+str(numCorrectPositive)+"("+str(numCorrectPositivePercent)+"%)")
    print("Total false positives: "+str(numFalsePositive)+"("+str(numFalsePositivePercent)+"%)")
    print("Total total false negatives: "+str(numFalseNegative)+"("+str(numFalseNegativePercent)+"%)")
    print("Total correct negatives: "+str(numCorrectNegative)+"("+str(numCorrectNegativePercent)+"%)")
    print()
    print("Recall measure:"+str(recall))
    print("Precision measure:"+str(precision))
    print("F-measure:"+str(fmeasure))