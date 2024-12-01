import os  # Module for interacting with the operating system (e.g., file paths).
import matplotlib.pyplot as plt  # Library for plotting graphs and images.
import matplotlib.image as mplimg  # Module to handle images in matplotlib.
import networkx as nx  # Library for creating and manipulating graphs and networks.
import random  # Module to perform random operations.

from io import BytesIO  # Provides an in-memory binary stream for image manipulation.
from itertools import chain  # Allows chaining multiple iterables into a single sequence.
from collections import namedtuple, OrderedDict  # Provides lightweight object structures.

# Define a namedtuple to represent a sentence with words and corresponding tags.
Sentence = namedtuple("Sentence", "words tags")


def read_data(filename: str) -> OrderedDict:
    """
    Reads tagged sentence data from a file and returns an ordered dictionary.

    Parameters:
        filename (str): Path to the file containing tagged sentence data.

    Returns:
        OrderedDict: A dictionary mapping sentence identifiers to Sentence objects.
    """
    with open(filename, 'r') as f:  # Open the file in read mode.
        # Split the file into sentences using double newlines and extract each line.
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    # Create an OrderedDict with sentence IDs as keys and Sentence objects as values.
    return OrderedDict(
        (
            s[0],  # Sentence ID as the key.
            Sentence(*zip(*[l.strip().split("\t") for l in s[1:]]))  # Words and tags as tuples.
        )
        for s in sentence_lines if s[0]  # Include only sentences with a valid ID.
    )


def read_tags(filename: str) -> frozenset:
    """
    Reads a list of word tag classes from a file.

    Parameters:
        filename (str): Path to the file containing tag data.

    Returns:
        frozenset: A set of unique tags.
    """
    with open(filename, 'r') as f:  # Open the file in read mode.
        tags = f.read().split("\n")  # Split the file into lines to extract tags.
    return frozenset(tags)  # Return a frozen set of unique tags.


def model2png(model, filename: str = "", overwrite: bool = False, show_ends: bool = False):
    """
    Converts a Pomegranate model into a PNG image.

    Parameters:
        model: A Pomegranate model with a .graph attribute (NetworkX graph).
        filename (str, optional): Path to save the PNG file. Defaults to "".
        overwrite (bool, optional): If True, overwrites an existing file. Defaults to False.
        show_ends (bool, optional): If True, includes start and end states in the graph. Defaults to False.

    Returns:
        np.ndarray: Image data of the graph as a PNG.
    """
    nodes = model.graph.nodes()  # Retrieve all nodes in the model graph.
    if not show_ends:  # Exclude start and end states if `show_ends` is False.
        nodes = [n for n in nodes if n not in (model.start, model.end)]
    # Create a subgraph with renamed nodes for readability.
    g = nx.relabel_nodes(model.graph.subgraph(nodes), {n: n.name for n in model.graph.nodes()})
    pydot_graph = nx.drawing.nx_pydot.to_pydot(g)  # Convert the NetworkX graph to a PyDot graph.
    pydot_graph.set_rankdir("LR")  # Set the graph direction to left-to-right.
    png_data = pydot_graph.create_png(prog='dot')  # Generate PNG data using Graphviz's dot tool.
    img_data = BytesIO()  # Create an in-memory binary stream.
    img_data.write(png_data)  # Write the PNG data to the stream.
    img_data.seek(0)  # Reset the stream position to the start.
    if filename:  # If a filename is provided:
        if os.path.exists(filename) and not overwrite:  # Check if file exists and overwrite is False.
            raise IOError("File already exists. Use overwrite=True to replace it.")
        with open(filename, 'wb') as f:  # Open the file in write-binary mode.
            f.write(img_data.read())  # Save the PNG data to the file.
        img_data.seek(0)  # Reset the stream position again.
    return mplimg.imread(img_data)  # Return the PNG image data.


def show_model(model, figsize: tuple = (5, 5), **kwargs):
    """
    Displays a Pomegranate model as an image using matplotlib.

    Parameters:
        model: A Pomegranate model with a .graph attribute (NetworkX graph).
        figsize (tuple, optional): Size of the matplotlib figure. Defaults to (5, 5).
        **kwargs: Additional arguments passed to `model2png`.
    """
    plt.figure(figsize=figsize)  # Create a new figure with specified dimensions.
    plt.imshow(model2png(model, **kwargs))  # Display the model as an image.
    plt.axis('off')  # Turn off axes for cleaner visualization.


class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    """
    Represents a subset of the dataset, containing word/tag sequences,
    vocabulary, and other metadata.
    """

    def __new__(cls, sentences: dict, keys: list):
        """
        Creates a new Subset instance.

        Parameters:
            sentences (dict): Dictionary mapping sentence IDs to Sentence objects.
            keys (list): List of sentence IDs included in this subset.

        Returns:
            Subset: A new instance of the Subset class.
        """
        word_sequences = tuple([sentences[k].words for k in keys])  # Collect word sequences.
        tag_sequences = tuple([sentences[k].tags for k in keys])  # Collect tag sequences.
        wordset = frozenset(chain(*word_sequences))  # Create a set of unique words.
        tagset = frozenset(chain(*tag_sequences))  # Create a set of unique tags.
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))  # Count total words.
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))  # Pair words with tags.
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__())

    def __len__(self) -> int:
        """Returns the number of sentences in the subset."""
        return len(self.sentences)

    def __iter__(self):
        """Returns an iterator over the sentences in the subset."""
        return iter(self.sentences.items())


class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    """
    Represents the entire dataset, including training and testing subsets.
    """

    def __new__(cls, tagfile: str, datafile: str, train_test_split: float = 0.8, seed: int = 112890):
        """
        Creates a new Dataset instance.

        Parameters:
            tagfile (str): Path to the file containing tag definitions.
            datafile (str): Path to the file containing sentences and tags.
            train_test_split (float, optional): Proportion of data for training. Defaults to 0.8.
            seed (int, optional): Random seed for reproducibility. Defaults to 112890.

        Returns:
            Dataset: A new instance of the Dataset class.
        """
        tagset = read_tags(tagfile)  # Load the set of tags.
        sentences = read_data(datafile)  # Load sentences with tags.
        keys = tuple(sentences.keys())  # Extract all sentence IDs.
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))  # Unique words in the dataset.
        word_sequences = tuple([sentences[k].words for k in keys])  # Sequence of words.
        tag_sequences = tuple([sentences[k].tags for k in keys])  # Sequence of tags.
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))  # Total word count.

        # Split dataset into training and testing sets.
        _keys = list(keys)
        if seed is not None:
            random.seed(seed)  # Set random seed for reproducibility.
        random.shuffle(_keys)  # Shuffle sentence IDs randomly.
        split = int(train_test_split * len(_keys))  # Determine split index.
        training_data = Subset(sentences, _keys[:split])  # Training subset.
        testing_data = Subset(sentences, _keys[split:])  # Testing subset.
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))  # Pair words with tags.
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__())

    def __len__(self) -> int:
        """Returns the total number of sentences in the dataset."""
        return len(self.sentences)

    def __iter__(self):
        """Returns an iterator over the sentences in the dataset."""
        return iter(self.sentences.items())
