import pandas as pd
from shallow import shallowLearn
from lstm import deepLearn
import numpy as np
import sys, getopt
import os
import time
loc = os.path.dirname(os.path.realpath(__file__))

# Constants
FILENAME = loc+"/news_ds.csv"
MODE = "shallow"
NUMBER_OF_TEST_ARTICLES = 500
VOCAB_SIZE = 7500
NOISY = False

# Shallow learning
USE_TDIDF = True
USE_NGRAMS = True

# Deep learning
USE_DROPOUT = True
DROPOUT_RATE = 0.1
EPOCHS = 2
BATCH_SIZE = 32
USE_RNN = False

def loadArticles(name):
    return pd.read_csv(name, header=0, delimiter=",")


def main():
    articles = loadArticles(FILENAME)
    t0 = time.time()
    if MODE == "shallow":
        print("Shallow learning with params:")
        print("NUMBER_OF_TEST_ARTICLES:",NUMBER_OF_TEST_ARTICLES," VOCAB_SIZE:",VOCAB_SIZE," USE_TDIDF:",USE_TDIDF," USE_NGRAMS:",USE_NGRAMS)
        shallowLearn(articles, NUMBER_OF_TEST_ARTICLES, VOCAB_SIZE, USE_TDIDF, USE_NGRAMS, NOISY=NOISY)
    elif MODE == "deep":
        print("Deep learning with params:")
        print("NUMBER_OF_TEST_ARTICLES:",NUMBER_OF_TEST_ARTICLES," VOCAB_SIZE:",VOCAB_SIZE," BATCH_SIZE:",BATCH_SIZE," USE_DROPOUT:",USE_DROPOUT, " USE_RNN:",USE_RNN)
        deepLearn(
            articles,
            NUMBER_OF_TEST_ARTICLES,
            EPOCHS = EPOCHS,
            BATCH_SIZE = BATCH_SIZE,
            VOCAB_SIZE = VOCAB_SIZE,
            USE_DROPOUT = USE_DROPOUT,
            DROPOUT_RATE = DROPOUT_RATE,
            USE_RNN = USE_RNN
        )
    t1 = time.time()
    total = t1-t0
    print()
    print("Time: ",total)


if __name__ == "__main__":

    # Parse options
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:v:ingradoe:b:")
    except getopt.GetoptError as e:
        print (str(e))
        print("Usage: "+sys.argv[0]+" -f /filename -v 7500 -i -n")
        print("-f <filename> Open filename as for the data")
        print("-v <number> Designate a vocab size as the argument")
        print("-n Enable noisy (debug) mode")
        print("Options for shallow:")
        print("    -a Use all terms as vocab (no limit)")
        print("    -i Turn off TF_IDF")
        print("    -g Turn off Ngrams")
        print("Options for deep:")
        print("    -d Use deep learning rather than shallow learning")
        print("    -o Turn off the dropout layer")
        print("    -e <number> Set the number of epochs to run for")
        print("    -b <number> Set the batch size")
        print("    -r Use RNN rather than LSTM")
        sys.exit(2)
    
    for o, a in opts:
        if o == '-n':
            NOISY = True
        if o == '-f':
            FILENAME = loc+a
        if o == '-v':
            VOCAB_SIZE = int(a)
        if o == '-g':
            USE_NGRAMS = not USE_NGRAMS
        if o == '-i':
            USE_TDIDF = not USE_TDIDF
        if o == '-a':
            VOCAB_SIZE = None
        if o == '-d':
            MODE = "deep"
            NUMBER_OF_TEST_ARTICLES = int(np.floor(NUMBER_OF_TEST_ARTICLES / 2))
        if o == '-o':
            USE_DROPOUT = not USE_DROPOUT
        if o == '-e':
            EPOCHS = int(a)
        if o == '-b':
            BATCH_SIZE = int(a)
        if o == '-r':
            USE_RNN = not USE_RNN
    
    main()