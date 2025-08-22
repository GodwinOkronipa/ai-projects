import json
import numpy as np
import nltk
import pickle
import random
import os
from nltk.stem.porter import PorterStemmer

# Initialize the stemmer
stemmer = PorterStemmer()

# Get the directory where this script is located
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if __file__ is not defined
    script_dir = os.path.dirname(os.path.abspath('chatbot.py'))

# Loading the Saved Files
with open(os.path.join(script_dir, 'intents.json'), 'r') as f:
    intents = json.load(f)
with open(os.path.join(script_dir, 'chatbot_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(script_dir, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)
with open(os.path.join(script_dir, 'tags.pkl'), 'rb') as f:
    tags = pickle.load(f)

# Preprocessing for User Input 
def tokenize_and_stem(word_list):
    ignore_words = ['?', '!', '.', ',']
    return [stemmer.stem(w.lower()) for w in word_list if w not in ignore_words]

# Main Chatbot Function
def get_response(user_input):
    # Preprocess the user's message
    tokenized_input = nltk.word_tokenize(user_input)
    stemmed_input = [' '.join(tokenize_and_stem(tokenized_input))]
    
    # Vectorize the user's input using the saved vectorizer
    input_vectorized = vectorizer.transform(stemmed_input)
    
    # Get the model's prediction
    prediction = model.predict(input_vectorized)
    predicted_tag_index = prediction[0]
    predicted_tag = tags[predicted_tag_index]
    
    # Get the list of responses for the predicted intent
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    
    return "I'm sorry, I don't understand that. Can you rephrase?"

#Main Conversation Loop 
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_message = input("You: ")
    if user_message.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break
    
    response = get_response(user_message)
    print("Chatbot:", response)