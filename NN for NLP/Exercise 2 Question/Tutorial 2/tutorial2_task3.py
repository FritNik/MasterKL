import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split

from tutorial2_task1 import load_glove_model


def generate_mappings(embeddings):
    """
    Generate mappings between words and unique integer IDs based on the provided embeddings.

    This function takes a dictionary of word embeddings and creates two mapping dictionaries:
    - One mapping words to their corresponding integer IDs (word2id).
    - The other mapping integer IDs back to the corresponding words (id2word).

    A special token `<unk>` for unknown words is also added to the mappings with an ID of 0. 
    This token is assigned a zero vector of the same dimension as the embeddings in the input dictionary.

    Parameters:
    embeddings (dict): A dictionary where keys are words (str) and values are their corresponding 
                       embeddings (numpy arrays).

    Returns:
    tuple: A tuple containing three elements:
        - vocabulary (list of str): A list of all words in the embeddings dictionary.
        - word2id (dict): A dictionary mapping words (str) to their unique integer IDs (int).
        - id2word (dict): A dictionary mapping integer IDs (int) to their corresponding words (str).

    Example:
    >>> embeddings = {"the": np.array([1.0, 2.0]), "a": np.array([3.0, 4.0])}
    >>> vocabulary, word2id, id2word = generate_mappings(embeddings)
    >>> print(vocabulary) # Prints ['<unk>', 'the', 'a']
    >>> print(word2id)   # Prints {'<unk>': 0, 'the': 1, 'a': 2}
    >>> print(id2word)   # Prints {0: '<unk>', 1: 'the', 2: 'a'}
    """
    word2id = {}
    id2word = {}
    vocabulary = ["<unk>"] + list(embeddings.keys())

    word2id["<unk>"] = np.zeros(len(embeddings["the"]), dtype=np.float32)
    id2word[0] = "<unk>" 
    
    for index, token in enumerate(vocabulary):
        word2id[token] = index + 1
        id2word[index + 1] = token
    
    return vocabulary, word2id, id2word


def tokenize(raw_text):
    """
    Tokenize a given string into individual words.

    This function preprocesses the raw text by replacing contractions of the form "n't" with " not" 
    to standardize them. It then tokenizes the modified text into individual words. The tokenization 
    is case-insensitive as the text is converted to lowercase before tokenizing.

    Parameters:
    raw_text (str): The raw text string that needs to be tokenized.

    Returns:
    list: A list of word tokens extracted from the input text.

    Note:
    The function uses regular expressions for tokenization, thus it captures words formed of alphanumeric 
    characters and ignores punctuation.

    Example:
    >>> text = "I can't believe it's not butter!"
    >>> tokenize(text)
    ['i', 'ca', 'not', 'believe', 'it', 's', 'not', 'butter']
    """
    # Replace "n't" with " not"
    text = re.sub(r"n't\b", " not", raw_text)

    # Convert to lowercase
    text = text.lower()

    # Extract alphanumeric word tokens
    tokens = re.findall(r"[a-z0-9]+", text) #characters a-z and digits 0-9 are allowed

    return tokens


def load_training_data(fname):
    """
    Load training data for sentiment analysis from a specified file.

    This function opens a file, reads its content line by line, and tokenizes each line. 
    It assumes that the file contains text data where each line represents a different review. 
    The function also creates a target tensor for sentiment labels, assigning 1 for positive 
    sentiment (which are assumed to be on the odd lines of the file) and 0 for negative sentiment 
    (on the even lines).

    Parameters:
    fname (str): The filename of the text file to be read.

    Returns:
    tuple: A tuple containing two elements:
        - words (list of list of str): A list where each element is a list of tokens 
          from each line of the file.
        - targets (torch.Tensor): A 1D tensor of the same length as `words`, containing 
          sentiment labels (1 for positive, 0 for negative).

    Example:
    >>> words, targets = load_training_data("reviews.txt")
    >>> print(words[0]) # Prints the tokens of the first line in the file
    >>> print(targets[0]) # Prints the sentiment label of the first line
    """
    words = []
    targets = []

    with open(fname, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tokens = tokenize(line.strip()) #delete leading white spaces and Zeilenumbruch
            words.append(tokens)

            # odd line number -> positive sentiment (1)
            # even line number -> negative sentiment (0)
            label = 1 if idx % 2 == 1 else 0
            targets.append(label)

    return words, torch.tensor(targets, dtype=torch.long)


def encode_and_pad(words, word2id, max_size):
    """
    Encode and pad a list of sentences to a specified maximum size.

    This function takes a list of sentences, where each sentence is a list of words. 
    It encodes each word into its corresponding integer ID using the provided `word2id` mapping. 
    If a word is not found in the mapping, it defaults to 0 (usually representing an unknown word). 
    Each sentence is then padded with zeros to ensure that all encoded sentences have the same length, 
    specified by `max_size`.

    If a sentence is longer than `max_size`, it is truncated to fit.

    Parameters:
    words (list of list of str): A list of sentences, each sentence being a list of words.
    word2id (dict): A dictionary mapping words (str) to their unique integer IDs (int).
    max_size (int): The maximum size to which each sentence will be padded or truncated.

    Returns:
    list: A list of 1D tensors, each tensor representing an encoded and padded sentence.

    Example:
    >>> words = [["hello", "world"], ["this", "is", "a", "test"]]
    >>> word2id = {"hello": 1, "world": 2, "this": 3, "is": 4, "a": 5, "test": 6}
    >>> max_size = 5
    >>> encoded_padded_sentences = encode_and_pad(words, word2id, max_size)
    >>> print(encoded_padded_sentences)
    [tensor([1, 2, 0, 0, 0]), tensor([3, 4, 5, 6, 0])]
    """
    features = [
        torch.tensor([word2id.get(token, 0) for token in line]) for line in words
    ]
    
    features_padded = []
    
    for feature in features:
        if len(feature) > max_size:
            features_padded.append(
                feature[:max_size]
            )
        else:
            features_padded.append(
                F.pad(feature, (0, max_size - len(feature)))
            )

    return features_padded


def train(inputs, targets, num_epochs, embeddings):
    """
    Train a sentiment classifier using given inputs and targets.

    This function initializes a SentimentClassifier with the specified vocabulary size and 
    embedding dimension. It uses the Adam optimizer and binary cross-entropy loss for training. 
    The function iterates over the specified number of epochs, training the classifier with the 
    provided inputs and targets. The loss is calculated for each input-target pair, and the 
    parameters of the classifier are updated accordingly. The function also prints the loss at 
    each epoch and accumulates the total loss over all epochs.
    """
    vocab_size = len(embeddings)
    embedding_dim = embeddings[0].shape[0]

    clf = SentimentClassifier(vocab_size, embedding_dim)
    optimizer = optim.Adam(clf.parameters(), lr=0.01)
    loss_function = nn.BCELoss()
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0

        for x, y in zip(inputs, targets):
            # x ist eine Liste aus Token-IDs → wir brauchen einen Tensor
            x_tensor = torch.tensor(x, dtype=torch.long)
            # Forward pass
            output = clf(x_tensor)

            # y auf float setzen für BCELoss, sonst bekomme ich Fehler
            y_tensor = torch.tensor(float(y), dtype=torch.float)

            loss = loss_function(output, y_tensor)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
        losses.append(total_loss)
    
    return losses, clf


class SentimentClassifier(nn.Module):
    """
    A sentiment classifier based on a Long Short-Term Memory (LSTM) network. WIll classify ppostive ir begative sentiment.
    """

    def __init__(self, input_size, embedding_dim, hidden_dim=256, LSTM_layers_size=2):
        super(SentimentClassifier, self).__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_size, embedding_dim) #werden hier mittrtainiert
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim, num_layers=LSTM_layers_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        # input ist ein 1D-Tensor mit Wort-IDs, z.B. [3, 10, 5, ...]

        # 1) Wort-IDs in Embeddings durch lernbar EMbeddinglayer (Form: [seq_len, embedding_dim])
        embedded = self.embedding(input)

        # 2) Batch-Dimension hinzufügen (LSTM erwartet [batch, seq_len, emb_dim])
        embedded = embedded.unsqueeze(0) 

        # 3) LSTM anwenden, letzer hidden_state repräsentiert den ganzen Satz
        _, (hidden_states, _) = self.lstm(embedded)

        # 4) Letzte LSTM-Schicht extrahieren (Form: [1, hidden_dim]), letzter hidden state beste repräsentiert den ganzen Satz
        last_hidden = hidden_states[-1]

        # 5) Sigmoid um WSL zu kriegen
        output = self.sig(self.fc(last_hidden))

        # 6) Form reduzieren → einzelner Wert
        return output.squeeze()

