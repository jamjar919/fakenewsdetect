import numpy as np
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from unidecode import unidecode

from collections import Counter

import nltk
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_vectors_web_lg')

np.random.seed(7)

VECTOR_DIMENSIONS = 300
MAX_SEQUENCE_LENGTH = 1000

# more constants
OTHER_TOKEN = "<OTHER>"
PAD_TOKEN = "<PAD>"

def textToVectors(text):
    words = text_to_word_sequence(text)
    vectors = []
    nonzero = 0
    for w in words:
        n = nlp(w)
        v = n.vector
        assert (len(v) == VECTOR_DIMENSIONS)
        if np.sum(v) > 0:
            nonzero += 1
            vectors.append(v)
        else:
            vectors.append(np.zeros(VECTOR_DIMENSIONS))
    # PAD and OTHER
    vectors.append(np.zeros(VECTOR_DIMENSIONS))
    vectors.append(np.zeros(VECTOR_DIMENSIONS))
    vectors.append(np.zeros(VECTOR_DIMENSIONS))
    print("Found",nonzero," nonzero vectors out of ",len(vectors)," (",100*nonzero/len(vectors),"% )")
    return vectors

def cleanText(dirty):
    # Convert to lowercase and remove punctuation
    dirty = unidecode(str(dirty))
    dirty = dirty.replace("\"", " ")
    clean = text_to_word_sequence(dirty)
    clean = " ".join(clean)
    return clean

def articleToInts(text, mapping):
    words = text_to_word_sequence(text)
    return [mapping[w] if w in mapping.keys() else mapping[OTHER_TOKEN] for w in words]

def intsToWords(ints, reverseMapping):
    words = [reverseMapping[i] for i in ints]
    return ' '.join(words)

def pad(sequence, padchar):
    if len(sequence) == MAX_SEQUENCE_LENGTH:
        return sequence
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        return sequence[0:MAX_SEQUENCE_LENGTH]
    else:
        return ([padchar]*(MAX_SEQUENCE_LENGTH - len(sequence)))+sequence

def deepLearn(articles, NUMBER_OF_TEST_ARTICLES=250, EPOCHS = 3, BATCH_SIZE = 64, VOCAB_SIZE = 10000, USE_DROPOUT=True, DROPOUT_RATE = 0.1, USE_RNN = False):
    # Split real and fake data
    real = articles[articles["LABEL"] == 1]
    fake = articles[articles["LABEL"] == 0]

    # Need to have equal sizes of real/fake data
    size = min(fake.shape[0], real.shape[0])
    real = real[0:size]
    fake = fake[0:size]

    # Get training and test sets of data
    fake_train, fake_test, real_train, real_test = train_test_split(fake, real, test_size=NUMBER_OF_TEST_ARTICLES, random_state=42)

    print("Collected",len(real_train),"real, ",len(fake_train),"fake train articles")
    print(len(real_test),"real, ",len(fake_test),"fake test articles")

    print("Cleaning text")
    texts = []
    target = []
    for a in real_train["TEXT"]:
        t = cleanText(a)
        texts.append(t)
        target.append(1)

    for a in fake_train["TEXT"]:
        t = cleanText(a)
        texts.append(t)
        target.append(0)

    # Get unique words
    allWords = ' '.join(texts)
    allWords = allWords.split(' ')
    uniqueWords = Counter(allWords).most_common()
    uniqueWords = uniqueWords[0:min(VOCAB_SIZE,len(uniqueWords))]
    uniqueWords = [k for k,v in uniqueWords]
    wordString = ' '.join(uniqueWords)
    print("Extracted",len(uniqueWords)," unique words")

    print("Getting spacy vectors")
    vectors = textToVectors(wordString)
    print("Got",len(vectors),"vectors")

    print("Encoding with one_hot")
    integers = one_hot(wordString, VOCAB_SIZE)

    # Make dicts so we can encode articles
    wordToInt = dict(zip(uniqueWords, integers))
    # Add other and pad
    wordToInt[OTHER_TOKEN] = len(integers) + 1
    wordToInt[PAD_TOKEN] = len(integers) + 2
    intToWord = dict(zip(wordToInt.values(), wordToInt.keys()))
    
    print("vectors:",len(vectors)," words:",len(wordToInt))


    # Change texts to integer representation
    print()
    for i in range(0, len(texts)):
        t = texts[i]
        intText = articleToInts(t, wordToInt)
        intText = pad(intText, wordToInt[PAD_TOKEN])
        if i == 0:
            print("Sanity check: First article")
            print(intText[-10:])
            print(intsToWords(intText, intToWord)[-100:])
        assert len(intText) == MAX_SEQUENCE_LENGTH
        texts[i] = intText
    print()

    # Np arrayify
    texts = np.array(texts)
    target = np.array(target)

    # Integerify test data
    test = []
    testTarget = []
    for a in real_test["TEXT"]:
        t = cleanText(a)
        t = pad(articleToInts(t, wordToInt), wordToInt[PAD_TOKEN])
        test.append(t)
        testTarget.append(1)

    for a in fake_test["TEXT"]:
        t = cleanText(a)
        t = pad(articleToInts(t, wordToInt), wordToInt[PAD_TOKEN])
        test.append(t)
        testTarget.append(0)

    # np arrayify again, but for test data
    test = np.array(test)
    testTarget = np.array(testTarget)

    print(texts)
    print(target)

    weights = np.asmatrix(vectors)
    print("Weights shape: ",weights.shape)

    model = Sequential()
    model.add(
        Embedding(
            input_dim=len(vectors),
            output_dim=VECTOR_DIMENSIONS,
            weights=[weights]
        )
    )
    if USE_RNN:
        model.add(SimpleRNN(64))
    else:
        model.add(LSTM(64))
    if USE_DROPOUT:
        model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    model.fit(x=texts, y=target, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test, testTarget))

    print("Testing...")
    pred = model.predict(test)

    # Set up evaluation vars
    numCorrectPositive = 0 # We classified as real, and it's real
    numFalsePositive = 0 # We classified as real, and it's fake 
    numFalseNegative = 0 # We classified as fake, and it's real
    numCorrectNegative = 0 # We classified as fake, and it's fake

    for i in range(0, len(pred)):
        p = pred[i]
        c = np.round(p)
        realClass = testTarget[i]
        # print("Model:",p," => (",c,") Actual:",realClass)
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
    numCorrectPositivePercent = 100 * (numCorrectPositive / len(test))
    numFalsePositivePercent = 100 * (numFalsePositive / len(test))
    numFalseNegativePercent = 100 * (numFalseNegative / len(test))
    numCorrectNegativePercent = 100 * (numCorrectNegative / len(test))
    totalCorrect = numCorrectPositive + numCorrectNegative
    totalCorrectPercent = 100 * (totalCorrect / len(test))
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

    

