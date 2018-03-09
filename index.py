import pandas as pd
from shallow import shallowLearn
import sys, getopt
import os
loc = os.path.dirname(os.path.realpath(__file__))

# Constants
FILENAME = loc+"/news_ds.csv"
# Shallow learning
NUMBER_OF_TEST_ARTICLES = 500
VOCAB_SIZE = 7500
USE_TDIDF = True
USE_NGRAMS = True


def loadArticles(name):
    return pd.read_csv(name, header=0, delimiter=",")


def main():
    articles = loadArticles(FILENAME)
    print("Shallow learning with params:")
    print("NUMBER_OF_TEST_ARTICLES:",NUMBER_OF_TEST_ARTICLES," VOCAB_SIZE:",VOCAB_SIZE," USE_TDIDF:",USE_TDIDF," USE_NGRAMS:",USE_NGRAMS)
    shallowLearn(articles, NUMBER_OF_TEST_ARTICLES, VOCAB_SIZE, USE_TDIDF, USE_NGRAMS)

if __name__ == "__main__":

    # Parse options
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:v:ina")
    except getopt.GetoptError as e:
        print (str(e))
        print("Usage: "+sys.argv[0]+" -f /filename -v 7500 -i -n")
        print("-v Designate a vocab size as the argument")
        print("-a Use all terms as vocab (no limit)")
        print("-i Turn off TF_IDF")
        print("-n Turn off Ngrams")
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

    
    main()