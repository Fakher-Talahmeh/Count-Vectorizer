import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# Download necessary NLTK data files
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the dataset
df = pd.read_csv('bbc_text_cls.csv')

# Separate features and labels
X = df['text']
y = df['labels']

# Visualize the distribution of labels
y.hist(figsize=(10, 5))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

# Define the function to get wordnet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Define the LemmaTokenizer class
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        
    def __call__(self, doc):
        tokens = word_tokenize(doc)
        words_and_tags = nltk.pos_tag(tokens)
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag)) 
                for word, tag in words_and_tags]

# Define the StemTokenizer class
class StemTokenizer:
    def __init__(self):
        self.porter = PorterStemmer()
        
    def __call__(self, doc):
        tokens = word_tokenize(doc)
        return [self.porter.stem(t) for t in tokens]

# Define a simple whitespace tokenizer
def simple_tokenizer(s):
    return s.split()
sentence=str(input("enter the sentence :"))
def predict_new_sentence(vectorizer_test,sentence):
    xNew = vectorizer_test.transform([sentence])
    prediction = model.predict(xNew)
    return prediction[0]


# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data and transform both training and test data
xTrain = vectorizer.fit_transform(X_train)
xTest = vectorizer.transform(X_test)

# Initialize the MultinomialNB model
model = MultinomialNB()

# Fit the model on the transformed training data
model.fit(xTrain, y_train)

# Print the training and test scores
print("train score: ", model.score(xTrain, y_train))
print("test score: ", model.score(xTest, y_test))
print("Predicted label for the new sentence:", predict_new_sentence(vectorizer,sentence))

# With stopwords
vectorizer = CountVectorizer(stop_words='english')
xTrain = vectorizer.fit_transform(X_train)
xTest = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(xTrain, y_train)
print("train score with stopwords: ", model.score(xTrain, y_train))
print("test score with stopwords: ", model.score(xTest, y_test))
print("Predicted label for the new sentence without stopwords:", predict_new_sentence(vectorizer,sentence))

# Using LemmaTokenizer
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
xTrain = vectorizer.fit_transform(X_train)
xTest = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(xTrain, y_train)
print("train score with LemmaTokenizer: ", model.score(xTrain, y_train))
print("test score with LemmaTokenizer: ", model.score(xTest, y_test))
print("Predicted label for the new sentence with LemmaTokeneizer:", predict_new_sentence(vectorizer,sentence))

# Using StemTokenizer
vectorizer = CountVectorizer(tokenizer=StemTokenizer())
xTrain = vectorizer.fit_transform(X_train)
xTest = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(xTrain, y_train)
print("train score with StemTokenizer: ", model.score(xTrain, y_train))
print("test score with StemTokenizer: ", model.score(xTest, y_test))
print("Predicted label for the new sentence with StemTokenizer:", predict_new_sentence(vectorizer,sentence))

# Using simple tokenizer
vectorizer = CountVectorizer(tokenizer=simple_tokenizer)
xTrain = vectorizer.fit_transform(X_train)
xTest = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(xTrain, y_train)
print("train score with simple tokenizer: ", model.score(xTrain, y_train))
print("test score with simple tokenizer: ", model.score(xTest, y_test))
print("Predicted label for the new sentence with simple tokenizer:", predict_new_sentence(vectorizer,sentence))
