#reads intents.json file, preprocesses the text, trains a Logistic Regression classifier(see ml-algorithms) and saves the trained model
import json
import numpy as np
import nltk
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure you have the required NLTK resources downloaded
try:
    _ = nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    _ = nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize the stemmer, this gives the most basic representation of the words
stemmer = PorterStemmer()

#Preprocessing Functions 
def tokenize_and_stem(word_list):
    ignore_words = ['?', '!', '.', ',']
    return [stemmer.stem(w.lower()) for w in word_list if w not in ignore_words]

def prepare_data():
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokenized_pattern = nltk.word_tokenize(pattern)
            all_words.extend(tokenized_pattern)
            xy.append((tokenized_pattern, tag))

    all_words = tokenize_and_stem(all_words)
    all_words = sorted(list(set(all_words)))
    tags = sorted(list(set(tags)))

    corpus = [' '.join(tokenize_and_stem(tokenized_pattern)) for tokenized_pattern, tag in xy]
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(corpus)

    y_train = []
    for _, tag in xy:
        label = [0] * len(tags)
        label[tags.index(tag)] = 1
        y_train.append(label)

    X_train = np.array(X_train_vectorized.toarray())
    y_train = np.array(y_train)

    return X_train, y_train, vectorizer, tags

#Training the Model 
if __name__ == "__main__":
    X_train, y_train, vectorizer, tags = prepare_data()

    print("Starting model training...")
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, np.argmax(y_train, axis=1))
    print("Model training complete.")

    #  Saving the Model and Vectorizer
    with open('chatbot_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('tags.pkl', 'wb') as f:
        pickle.dump(tags, f)
    
    print("All necessary files saved.")