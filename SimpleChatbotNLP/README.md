# SimpleChatbotNLP

A simple Natural Language Processing chatbot built with Python, using Logistic Regression for intent classification and NLTK for text preprocessing.

## Overview

This chatbot can handle basic conversations including greetings, farewells, thanks, questions about its creator, and gracefully deflects complex requests it cannot fulfill.

**Created by:** Godwin Okronipa as a side project

## Features

- **Intent Recognition**: Classifies user input into predefined categories
- **Natural Language Processing**: Uses NLTK for tokenization and stemming
- **Machine Learning**: Employs Logistic Regression with TF-IDF vectorization
- **Extensible**: Easy to add new intents and responses

## Project Structure

SimpleChatbotNLP/
 intents.json        # Training data with intents, patterns, and responses
 train_chatbot.py    # Model training script
 chatbot.py          # Main chatbot interface
 README.md           # This file


## Supported Intents

- **Greeting**: Hi, Hello, Hey, Good morning, etc.
- **Goodbye**: Bye, Goodbye, See you later, etc.
- **Thanks**: Thank you, Thanks, I appreciate it, etc.
- **Creator**: Who created you, Who made you, etc.
- **Actions**: Handles requests the bot cannot yet fulfill (stories, emails, calculations, etc.)

## Installation

1. **Install required packages:**
   ```bash
   pip install nltk numpy scikit-learn
   ```

2. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

1. **Train the model:**
   ```bash
   python train_chatbot.py
   ```
   This creates:
   - `chatbot_model.pkl` - Trained Logistic Regression model
   - `vectorizer.pkl` - TF-IDF vectorizer
   - `tags.pkl` - Intent tags

2. **Run the chatbot:**
   ```bash
   python chatbot.py
   ```

3. **Chat with the bot:**
   ```
   You: Hello
   Chatbot: Hi there! What can I do for you?
   
   You: Who created you?
   Chatbot: I was created by Godwin Okronipa as a side project!
   
   You: quit
   Chatbot: Goodbye!
   ```

## Technical Details

- **Algorithm**: Logistic Regression classifier (See ml-algorithms folder for more info)
- **Text Processing**: Porter Stemmer for word normalization
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Libraries**: NLTK, scikit-learn, NumPy

## Extending the Chatbot

To add new intents:

1. **Edit `intents.json`:**
   ```json
   {
     "tag": "new_intent",
     "patterns": ["pattern1", "pattern2"],
     "responses": ["response1", "response2"]
   }
   ```

2. **Retrain the model:**
   ```bash
   python train_chatbot.py
   ```

## Files Generated After Training

- `chatbot_model.pkl` - Serialized trained model
- `vectorizer.pkl` - Fitted TF-IDF vectorizer
- `tags.pkl` - List of intent tags

## Future Enhancements

- Add more sophisticated NLP techniques
- Implement context awareness
- Add voice capabilities (the bot mentions it will "sing very soon"!)
- Expand intent categories
- Add confidence scoring

## License

This is a personal side project by Godwin Okronipa.
