import pandas as pd
from shallow import shallowLearn
from lstm import deepLearn
import numpy as np
import sys, getopt
import os
loc = os.path.dirname(os.path.realpath(__file__))

# Constants
FILENAME = loc+"/news_ds.csv"
MODE = "shallow"
NUMBER_OF_TEST_ARTICLES = 500

# Shallow learning
VOCAB_SIZE = 7500
USE_TDIDF = True
USE_NGRAMS = True


def loadArticles(name):
    return pd.read_csv(name, header=0, delimiter=",")


def main():
    articles = loadArticles(FILENAME)
    if MODE == "shallow":
        print("Shallow learning with params:")
        print("NUMBER_OF_TEST_ARTICLES:",NUMBER_OF_TEST_ARTICLES," VOCAB_SIZE:",VOCAB_SIZE," USE_TDIDF:",USE_TDIDF," USE_NGRAMS:",USE_NGRAMS)
        shallowLearn(articles, NUMBER_OF_TEST_ARTICLES, VOCAB_SIZE, USE_TDIDF, USE_NGRAMS)
    elif MODE == "deep":
        deepLearn(articles, NUMBER_OF_TEST_ARTICLES)

if __name__ == "__main__":

    # Parse options
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:v:inad")
    except getopt.GetoptError as e:
        print (str(e))
        print("Usage: "+sys.argv[0]+" -f /filename -v 7500 -i -n")
        print("-f filename Open filename as for the data")
        print("Options for shallow:")
        print("    -v Designate a vocab size as the argument")
        print("    -a Use all terms as vocab (no limit)")
        print("    -i Turn off TF_IDF")
        print("    -n Turn off Ngrams")
        print("Options for deep:")
        sys.exit(2)
    
    for o, a in opts:
        if o == '-f':
            FILENAME = loc+a
        if o == '-v':
            VOCAB_SIZE = int(a)
        if o == '-n':
            USE_NGRAMS = not USE_NGRAMS
        if o == '-i':
            USE_TDIDF = not USE_TDIDF
        if o == '-a':
            VOCAB_SIZE = None
        if o == '-d':
            MODE = "deep"
            NUMBER_OF_TEST_ARTICLES = int(np.floor(NUMBER_OF_TEST_ARTICLES / 2))

    
    main()