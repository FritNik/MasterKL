import re

from matplotlib import pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os



def tokenize(fname):
    """
    Tokenizes the text from a file into words.

    This function reads the content of a file, performs basic preprocessing, and 
    then tokenizes the text into words. The tokenization process involves converting 
    the text to lowercase and splitting it into words based on word boundaries.

    Parameters:
    - fname (str): The path to the file containing the text to be tokenized.

    Returns:
    - list of str: A list of words extracted from the file.

    Note:
    - The function assumes the file is encoded in UTF-8.
    - Punctuation is ignored, and contractions like "n't" are expanded for consistency in tokenization.

    Example:
    >>> tokenize("example.txt")
    ['this', 'is', 'an', 'example', 'text']
    """
    with open(fname, 'r', encoding="utf8") as f:
        text = f.read().lower()
    
    text = re.sub(r"n't", " not", text)
    
    # Tokenization using regex 
    tokens = re.findall(r'\b\w+\b', text) #Boundary of a word \w+ one or more word characters \b boundary of a word
    
    return tokens





def generate_training_data(tokens, context_window_size):
    """
    Generate training data for word embedding models from a list of tokens.

    This function creates pairs of center words and their respective context words based on a given
    context size. For each token in the input list, it considers a fixed number of tokens before and 
    after it as its context. These pairs are useful for training certain types of word embedding models 
    such as Word2Vec.

    Parameters:
    - tokens (list of str): A list of tokens (words) from which to generate the training data.
    - context_window_size (int): The number of tokens to include before and after each center word as its context.

    Returns:
    - tuple of (list of str, list of list of str): A tuple where the first element is a list of center words 
      and the second element is a list of lists, each containing the context words corresponding to each center word.

    Note:
    - The function does not generate context for tokens at the beginning and end of the list where 
      sufficient context is not available (i.e., less than `context_window_size` tokens before or after).

    Example:
    >>> generate_training_data(["the", "quick", "brown", "fox", "jumps"], 2)
    (['brown'], [['the', 'quick', 'fox', 'jumps']])
    """
    center_words = []
    context_words = []
    for i in range(context_window_size, len(tokens) - context_window_size):
        center = tokens[i]
        context = tokens[i - context_window_size:i] + tokens[i + 1:i + 1 + context_window_size]
        center_words.append(center)
        context_words.append(context)
    
    return center_words, context_words


def generate_mappings(tokens):
    """
    Generate mapping dictionaries for word to ID and ID to word conversions.

    This function creates two dictionaries to map words to unique integer IDs and vice versa, 
    along with a set representing the vocabulary. It processes a list of tokens and assigns a 
    unique ID to each distinct word in the list. This is particularly useful for word embedding 
    and natural language processing tasks where words need to be converted to numeric forms.

    Parameters:
    - tokens (list of str): A list of tokens (words) for which the mappings will be generated.

    Returns:
    - tuple: A tuple containing three elements:
        1. vocabulary (set of str): A set of unique words found in the input tokens.
        2. word2id (dict): A dictionary mapping each word in the vocabulary to a unique integer ID.
        3. id2word (dict): A dictionary mapping each unique integer ID back to its corresponding word.

    Note:
    - The IDs in the dictionaries are assigned based on the order of words encountered in the set 
      created from the tokens, which might differ from their order in the original token list due to 
      the nature of sets in Python.

    Example:
    >>> generate_mappings(["hello", "world", "hello"])
    ({"hello", "world"}, {"hello": 0, "world": 1}, {0: "hello", 1: "world"})
    """
    word2id = {}
    id2word = {}
    vocabulary = set(tokens)
    
    for index, token in enumerate(vocabulary):
        word2id[token] = index
        id2word[index] = token
    
    return vocabulary, word2id, id2word

import torch
import torch.nn as nn
import torch.nn.functional as F


        

class CBOW(nn.Module):
    """
    A Continuous Bag of Words (CBOW) model implementation using PyTorch.

    This class implements the CBOW model, a type of neural network used for natural language processing. 
    It predicts a target word based on context words within a fixed window size. The model uses embeddings 
    to represent words and employs linear layers for prediction.
    """
  
    def __init__(self, vocab_size, context_size, embedding_dim):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128) #we get several context words to predict one word, so dimesnion of contextWindow times embedding_dim
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        output = None
        embeds = self.embeddings(inputs)            
        embeds = embeds.view(1, -1)    
        output = F.relu(self.linear1(embeds))
        output = self.linear2(output)
        
        return F.log_softmax(output, dim=-1)


class SkipGram(nn.Module):
    """
    A Skip-Gram model implementation using PyTorch.

    The Skip-Gram model is a type of neural network used in natural language processing for word embedding. 
    It aims to predict the context words from a target word, effectively learning word associations from 
    the surrounding context. The model consists of embedding and linear layers, and it utilizes ReLU 
    activation and log softmax for its output.
    """

    def __init__(self, vocab_size, context_size, embedding_dim):
        super(SkipGram, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128) #we want to say how likely a word out of vocab is to appear next to given center word, so one word is input
        self.linear2 = nn.Linear(128, context_size * vocab_size)
        
        self.context_size = context_size

    def forward(self, inputs):
        output = None
        embeds = self.embeddings(inputs)
        output = F.relu(self.linear1(embeds))
        output = self.linear2(output)

        return F.log_softmax(output, dim=-1)


def train_word2vec(model, inputs, targets, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_function = nn.NLLLoss()

    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for input, target in zip(inputs, targets):
            model.zero_grad()

            input_tensor = input
            target_tensor = target


            output = model.forward(input_tensor)
            loss = loss_function(output, target_tensor)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        losses.append(total_loss)
    return losses


def plot_loss(title, axis):
    """
    Plot loss curves for one or more training processes.

    This function takes a list of loss data from different training processes and plots them on the same graph. 
    Each set of loss data can be labeled and colored for differentiation. This is useful for visually comparing 
    training losses over epochs for different models or parameters.

    Parameters:
    - title (str): The title of the plot, describing what the graph represents.
    - axis (list of tuples): A list where each tuple contains three elements:
        1. loss (list of float): A list of loss values, typically one per epoch.
        2. label (str): A label for the loss data, used in the graph's legend.
        3. color (str): The color used for plotting this set of loss data.

    Returns:
    - None: This function does not return anything but shows a matplotlib plot.

    Example:
    >>> plot_loss("Training Loss Comparison", 
                  [(loss_model1, "Model 1", "blue"), 
                   (loss_model2, "Model 2", "red")])
    # This will plot the losses of Model 1 and Model 2 on the same graph for comparison.
    """
    _, ax = plt.subplots()

    for axis_i in axis:
        loss, label, color = axis_i
        ax.plot(loss, label=label, color=color)
    
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    plt.title(title)
    plt.show()
