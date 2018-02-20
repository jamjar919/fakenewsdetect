import pandas as pd 
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
nltk.download();


def cleanText(dirty):
    # Convert to lowercase and remove punctuation
    clean = text_to_word_sequence(article[1]);
    # Remove stopwords
    stops = set(stopwords.words("english"));
    clean = [w for w in clean if not w in stops]
    return (" ".join(clean));


articles = pd.read_csv("news_ds.csv", header=0, delimiter=",")

print(articles)

# Calculate base probabilities for P_fake, P_real (the chance of an article being classified into these categories)
numArticles = len(articles);
numFake = 0;
numReal = 0;

fakeArticles = list();
realArticles = list();

for i in range(0, numArticles):
    if (articles["TEXT"][i] == '1'):
        realArticles.append(article);
        numReal += 1;
    else:
        fakeArticles.append(article);
        numFake += 1;

P_fake = numFake / numArticles;
P_real = numReal / numArticles;

# Preprocess text in the articles using keras

# tokenize the document - this converts to lowercase and removes punctuation. Do this for all fake and all real articles
fakeWords = list();
print("cleaning fake")
for article in fakeArticles:
    fakeWords.extend(cleanText(article[1]));

realWords = list();
print("cleaning real")
for article in realArticles:
    realWords.extend(text_to_word_sequence(article[1]));

print(realWords);