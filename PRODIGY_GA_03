# Task3: Text Generation with Markov Chains

'''Problem Statement: Implement a simple text generation algorithm using Markov chains. 
This task involves creating a statistical model that predicts the probability 
of a character or word based on the previous ones(s).'''

import random
import re
import urllib.request

# Download the text corpus (Alice's Adventures in Wonderland)
url = "https://www.gutenberg.org/files/11/11-0.txt"
response = urllib.request.urlopen(url)
raw_text = response.read().decode('utf-8')

# Clean and preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Apply preprocessing to the raw text
cleaned_text = preprocess_text(raw_text)

# Tokenize the text by splitting it into words
tokens = cleaned_text.split()

# Build the Markov chain model
def build_markov_chain(tokens):
    # Initialize an empty dictionary for the Markov chain
    markov_chain = {}
    # Loop through the tokens to build the chain
    for i in range(len(tokens) - 1):
        word = tokens[i]
        next_word = tokens[i + 1]
        if word not in markov_chain:
            # Create a new list for the word if it doesn't exist in the chain
            markov_chain[word] = []
        # Append the next word to the list of the current word
        markov_chain[word].append(next_word)
    return markov_chain

# Create the Markov chain using the tokenized text
markov_chain = build_markov_chain(tokens)

# Generate text using the Markov chain model
def generate_text(markov_chain, start_word, length=100):
    current_word = start_word
    generated_text = [current_word]
    for _ in range(length - 1):
        if current_word not in markov_chain:
            break
        # Choose a random next word from the list of next words
        next_word = random.choice(markov_chain[current_word])
        generated_text.append(next_word)
        # Update the current word to the next word
        current_word = next_word
    return ' '.join(generated_text)

# Select a random start word from the tokens
start_word = random.choice(tokens)

# Generate text of specified length starting from the random start word
generated_text = generate_text(markov_chain, start_word, length=100)
print(generated_text)
