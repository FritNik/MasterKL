import numpy as np

from matplotlib import pyplot as plt


def load_glove_model(filename):
    """
    Following function returns pretrained glove model in the dictionary format.
    The keys will be all the words in vocabulary.
    The value will be the corresponding vector.
    """
    
    model = {}
    with open(filename, 'r', encoding="utf8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float32)
            model[word] = embedding
    return model


def cosine_distance(vec_a, vec_b):
    """
    Calculate the cosine distance between two vectors.

    This function computes the cosine distance, which is a measure of similarity between two non-zero vectors.
    The cosine distance is defined as 1 minus the cosine similarity between the vectors. Cosine similarity is
    the cosine of the angle between the two vectors, calculated as the dot product of the vectors divided by
    the product of their magnitudes.

    Parameters:
    - vec_a (array-like): The first vector.
    - vec_b (array-like): The second vector. It must be the same length as vec_a.

    Returns:
    - float: The cosine distance between vec_a and vec_b, ranging from 0 (identical) to 2 (opposite).

    Note:
    - The function assumes that both vectors are non-zero and have the same dimension.

    Example:
    >>> cosine_distance([1, 0, 0], [0, 1, 0])
    1.0  # The vectors are orthogonal, so the cosine distance is 1.
    """
    vec_a = np.array(vec_a) 
    vec_b = np.array(vec_b) 
    # Cosine similarity  measure 
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    cosine_similarity = dot_product / (norm_a * norm_b) 
    
    cosine_distance = 1 - cosine_similarity

    # Cosine distance cosine_distance = 1 - cosine_similarity 
    return cosine_distance

def euclidean_distance(vec_a, vec_b):
    """
    Calculate the Euclidean distance between two vectors.

    This function computes the Euclidean distance, which is the "straight-line" distance between two points
    in Euclidean space. In the context of vectors, it's the square root of the sum of the squared differences
    between corresponding elements of the two vectors.

    Parameters:
    - vec_a (array-like): The first vector. It can be a list, tuple, or any array-like structure.
    - vec_b (array-like): The second vector. It must be the same length as vec_a.

    Returns:
    - float: The Euclidean distance between vec_a and vec_b.

    Note:
    - Both vec_a and vec_b should be of the same dimensions.
    - This function does not check for the length of the vectors; non-matching lengths will result in an error.

    Example:
    >>> euclidean_distance([1, 2, 3], [4, 5, 6])
    5.196152422706632  # The calculated Euclidean distance between the two vectors.
    """
    vec_a = np.array(vec_a) 
    vec_b = np.array(vec_b) 
    distance = np.sum((vec_a - vec_b) ** 2) ** 0.5
    return distance

def most_similar(vec, model, distance_func, topk = 10, ignore_vec=np.array([])):
    """
    Find the top 'k' words most similar to a given vector within a specified model.

    This function computes the similarity of each word in the model to the given vector 'vec',
    using a specified distance or similarity function. It returns a list of the top 'k' similar
    words, sorted in order of increasing distance/similarity.

    Parameters:
    - vec (array-like): The reference vector for which similar words are to be found.
    - model (dict): A dictionary-like object where keys are words and values are their corresponding vectors.
    - distance_func (callable): A function to compute the similarity or distance between two vectors.
    - topk (int, optional): The number of top similar words to return. Defaults to 10.
    - ignore_vec (array-like, optional): An array of vectors to be ignored in the similarity calculation.
      Useful for excluding the input vector or other specific vectors from consideration.

    Returns:
    - list: A list of tuples, each containing a word and its similarity score, sorted by increasing similarity.

    Example:
    >>> model = {'apple': np.array([1, 2]), 'banana': np.array([2, 3]), 'orange': np.array([3, 4])}
    >>> vec = np.array([1.1, 1.9])
    >>> most_similar(vec, model, cosine_distance)
    [('apple', 0.0018689664347796286), ('banana', 0.0019968868522816097), ('orange', 0.0070372092900762295)]
    """
    vec = np.array(vec) 
    ignore_vecs = [np.array(v) for v in ignore_vec] if ignore_vec is not None else []
    results = [] 
    for word, word_vec in model.items():
        word_vec = np.array(word_vec) 
        if any(np.array_equal(word_vec, iv) for iv in ignore_vecs):
            continue
        distance = distance_func(vec, word_vec)
        results.append((word, distance))
  
    results.sort(key=lambda x: x[1]) 
    return results[:topk]


def show_similarities(words, model, distance_func):
    """
    Display a heatmap to visualize the similarities between pairs of words.

    This function computes the similarity between each pair of words in the provided list 
    using the specified distance function. It then displays these similarities in a heatmap,
    allowing for an intuitive visual comparison of how closely related each pair of words is
    within the given model's embedding space.

    Parameters:
    - words (list of str): A list of words to compare. Each word should be present in the model.
    - model (dict): A dictionary-like object mapping words to their vector embeddings.
    - distance_func (callable): A function to compute the distance or similarity between two vectors.

    Returns:
    - None: The function does not return anything but displays a heatmap plot.

    Note:
    - The function assumes all words in `words` are present in `model`.
    - `distance_func` should take two vectors as input and return a scalar representing their distance or similarity.

    Example:
    >>> show_similarities(['cat', 'dog', 'fish'], word_embeddings, cosine_distance)
    # Displays a heatmap showing the similarities between 'cat', 'dog', and 'fish'.
    """
    n_words = len(words)
    sim = []
    for j in words:
        temp = []
        for k in words:
            temp.append(distance_func(model[j], model[k]))
        sim.append(temp)
    data = np.array(sim)
    data.resize(n_words, n_words)

    plt.figure(figsize = (4, 4))
    plt.imshow(data, interpolation='nearest')
    plt.xticks(range(n_words), words, rotation=90)
    plt.yticks(range(n_words), words)
    plt.show()


def analogies(word1, word2, word3, model):
    """
    Compute the word that forms an analogy with the given three words in the specified model.

    This function finds a word such that 'word1' is to 'word2' as 'word3' is to this unknown word.
    In terms of word embeddings, it computes this by finding the nearest vector in the model to
    the result of vec_word2 - vec_word1 + vec_word3.

    The function internally uses 'most_similar' to find the nearest word to this computed vector,
    while ignoring the vectors of 'word1', 'word2', and 'word3' in the similarity search.

    Parameters:
    - word1 (str): The first word in the analogy.
    - word2 (str): The second word in the analogy, related to 'word1'.
    - word3 (str): The third word in the analogy, seeking a word related to it as 'word2' is to 'word1'.
    - model (dict): A dictionary-like object mapping words to their vector embeddings.

    Returns:
    - str: The word that completes the analogy with 'word1', 'word2', and 'word3'.

    Example:
    >>> model = {'king': np.array([...]), 'man': np.array([...]), 'woman': np.array([...]), ...}
    >>> analogies('man', 'king', 'woman', model)
    'queen'  # assuming 'queen' is the nearest vector result
    """
    vec = model[word2] - model[word1] + model[word3]
    similar_words = most_similar(vec, model, cosine_distance, ignore_vec=np.array([model[word1], model[word2], model[word3]]))
    
    return similar_words[0][0]
