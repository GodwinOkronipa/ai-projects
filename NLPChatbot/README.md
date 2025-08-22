# SimpleChatbotNLP

A Natural Language Processing chatbot built with Python, using Logistic Regression for intent classification and NLTK for text preprocessing. Now enhanced with **Machine Learning knowledge** to answer questions about ML concepts, algorithms, and applications

## Overview

This intelligent chatbot can handle conversations about:
- Basic interactions (greetings, farewells, thanks)
- Questions about its creator
- **Machine Learning topics** (algorithms, concepts, applications)
- Data Science fundamentals

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

### Basic Interactions
- **Greeting**: Hi, Hello, Hey, Good morning, etc.
- **Goodbye**: Bye, Goodbye, See you later, etc.
- **Thanks**: Thank you, Thanks, I appreciate it, etc.
- **Creator**: Who created you, Who made you, etc.

### Machine Learning & Data Science
- **ML Basics**: What is machine learning, How does ML work, etc.
- **ML Types**: Supervised vs unsupervised learning, Types of ML, etc.
- **Supervised Learning**: Classification, regression, labeled data, etc.
- **Unsupervised Learning**: Clustering, pattern recognition, etc.
- **Algorithms**: Popular ML algorithms, neural networks, decision trees, etc.
- **Deep Learning**: Neural networks, deep vs machine learning, etc.
- **ML Applications**: Real-world uses, examples, where ML is used, etc.
- **Data Science**: Data science vs ML, data scientist role, etc.

### General
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
   
   You: What is machine learning?
   Chatbot: Machine Learning is a subset of AI that enables computers to learn and make decisions from data without being explicitly programmed for every task.
   
   You: Types of machine learning
   Chatbot: The main types are: Supervised Learning (with labeled data), Unsupervised Learning (finding patterns in unlabeled data), and Reinforcement Learning (learning through rewards and penalties).
   
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
- Expand intent categories
- Add confidence scoring

## License

This is a personal side project by Godwin Okronipa.
