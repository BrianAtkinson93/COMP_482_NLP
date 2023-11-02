import os
import sys
import tarfile
import zipfile
import pickle
import PyPDF2

from typing import Tuple, List, Any, Dict
from collections import Counter
from gensim.models import Word2Vec
from nltk import word_tokenize
from tqdm import tqdm
from numpy import ndarray


def compress_with_word2vec(tokens: List[str]) -> Tuple[list[str], Word2Vec]:
    """
    Compresses text using Word2Vec model.

    :param tokens: List of word tokens from the text.
    :return: Compressed text and the Word2Vec model used for compression.
    """
    model = Word2Vec(sentences=[tokens], vector_size=100, window=5, min_count=1, sg=0)
    compressed_text = [model.wv[word] for word in tokens if word in model.wv.index_to_key]
    return compressed_text, model


def decompress_with_word2vec(compressed_text: List[List[float]], model: Word2Vec) -> List[Any]:
    """
    Decompresses text compressed using a Word2Vec model.

    :param compressed_text: List of compressed tokens.
    :param model: The Word2Vec model used for compression.
    :return: Decompressed text.
    """
    decompressed_text = [model.wv.most_similar([vector], topn=1)[0][0] for vector in compressed_text]
    return decompressed_text


def get_file_size(file_path, units='MB') -> float:
    """
    Gets the file size in specified units.

    :param file_path: Path to the file.
    :param units: Units in which size is to be returned ('KB', 'MB', 'GB').
    :return: File size in specified units.
    """
    size_in_bytes = os.path.getsize(file_path)
    if units == 'KB':
        return size_in_bytes / 1024
    elif units == 'MB':
        return (size_in_bytes / 1024) / 1024
    elif units == 'GB':
        return ((size_in_bytes / 1024) / 1024) / 1024
    else:
        return size_in_bytes


def compress_with_frequency(tokens: List[str]) -> tuple[list[Any], dict]:
    """
    Compresses text using frequency-based compression.

    :param tokens: List of word tokens from the text.
    :return: Compressed text and a dictionary mapping words to their frequency-based index.
    """

    # Create a dictionary sorted by frequency
    word_freq = Counter(tokens)
    sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)
    word_to_index = {word: index for index, word in enumerate(sorted_words)}

    # Replace each word with its index
    compressed_text = [word_to_index[word] for word in tokens]

    return compressed_text, word_to_index


def decompress_with_frequency(compressed_text: List[str], word_to_index: Dict[str, str]) -> List[list[Any]]:
    """
    Decompresses text that was compressed using frequency-based compression.

    :param compressed_text: List of compressed tokens.
    :param word_to_index: Dictionary mapping words to their frequency-based index.
    :return: Decompressed text.
    """

    # Reverse the word_to_index dictionary
    index_to_word = {index: word for word, index in word_to_index.items()}

    # Replace each index with its word
    decompressed_text = [index_to_word[index] for index in compressed_text]

    return decompressed_text


def read_pdf(file_path: str) -> List[str]:
    """
    Reads a PDF file and tokenizes its content.

    :param file_path: Path to the PDF file.
    :return: List of word tokens from the PDF file.
    """
    tokens = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)
        for i in tqdm(range(total_pages), desc="Reading PDF pages", unit="page"):
            page = pdf_reader.pages[i]
            text = page.extract_text()
            tokens.extend(word_tokenize(text.lower()))
    return tokens


def read_txt(file_path: str) -> List[str]:
    """
    Reads a text file and tokenizes its content.

    :param file_path: Path to the text file.
    :return: List of word tokens from the text file.
    """
    tokens = []
    total_lines = sum(1 for line in open(file_path, 'r', encoding='utf-8'))
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Reading lines", unit="line"):
            try:
                tokens.extend(word_tokenize(line.lower()))
            except UnicodeDecodeError:
                continue
    return tokens


def save_compressed_data(compressed_text: List[str], file_name: str) -> None:
    """
    Saves compressed text to a file.

    :param compressed_text: List of compressed tokens.
    :param file_name: Name of the file to save the compressed text.
    :return: None
    """
    with open(file_name, 'wb') as f:
        pickle.dump(compressed_text, f)


def load_compressed_data(file_name: str) -> List[Any]:
    """
    Loads compressed text from a file.

    :param file_name: Name of the file from which to load the compressed text.
    :return: Loaded compressed text.
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def load_model(model_name: str) -> Word2Vec:
    """
    Loads a Word2Vec model from a file.

    :param model_name: The name of the file from which to load the model.
    :return: Loaded Word2Vec model.
    """
    return Word2Vec.load(model_name)


def save_model(model: Word2Vec, model_name: str):
    """
    Saves a Word2Vec model to a file.

    :param model: The Word2Vec model to save.
    :param model_name: The name of the file to save the model.
    """
    model.save(model_name)


def display_file_sizes(original_file, compressed_file) -> None:
    """
    Displays the sizes of the original and compressed files in MB.

    :param original_file: Path to the original file.
    :param compressed_file: Path to the compressed file.
    """
    original_size = get_file_size(original_file)
    compressed_size = get_file_size(compressed_file)
    print(f"Size of original file: {original_size} MB")
    print(f"Size of compressed file: {compressed_size} MB")


def create_and_display_archives(original_file, compressed_file) -> None:
    """
    Creates ZIP and tar.gz archives for the original and compressed files and displays their sizes.

    :param original_file: Path to the original file.
    :param compressed_file: Path to the compressed file.
    """
    with zipfile.ZipFile(f'{original_file}.zip', 'w') as zipf:
        zipf.write(original_file)
    with tarfile.open(f'{original_file}.tar.gz', 'w:gz') as tar:
        tar.add(original_file)
    zip_size = get_file_size(f'{original_file}.zip')
    tar_size = get_file_size(f'{original_file}.tar.gz')
    compressed_size = get_file_size(compressed_file)
    print(f"Size of ZIP file: {zip_size} MB, diff: {compressed_size - zip_size}")
    print(f"Size of tar.gz file: {tar_size} MB, diff: {compressed_size - tar_size}")


def frequency_based(tokens: List[str], file_path: str) -> None:
    """
    Frequency based approach containment.

    :param tokens: List of word tokens from the text.
    :param file_path: Path to the file.
    :return: None
    """
    print('\n' + '*' * 25)
    print('Running the frequency based approach...')
    compressed_text, word_to_index = compress_with_frequency(tokens)
    save_compressed_data(compressed_text, 'compressed_data.pkl')
    loaded_compressed_text = load_compressed_data('compressed_data.pkl')
    decompressed_text = decompress_with_frequency(loaded_compressed_text, word_to_index)
    print("Decompression successful, original text recovered.\n") if tokens == decompressed_text else print(
        "Decompression failed.\n")

    # Display Freq-based comparison
    display_file_sizes(file_path, 'compressed_data.pkl')
    create_and_display_archives('original_data.txt', 'compressed_data.pkl')


def word2vec(tokens: List[str], file_path: str) -> None:
    """
    Word2Vec based containment.

    :param tokens: List of word tokens from the text.
    :param file_path: Path to the file.
    :return: None
    """
    print('\n' + '*' * 25)
    # Word2Vec-based compression
    print('Running the compression model with word2vec...')
    compressed_text, model = compress_with_word2vec(tokens)
    save_compressed_data(compressed_text, 'compressed_data_model.pkl')

    # Save the model
    model_name = "word2vec.model"
    save_model(model, model_name)

    # Reload the model
    loaded_model = load_model(model_name)
    loaded_compressed_text = load_compressed_data('compressed_data_model.pkl')
    decompressed_text = decompress_with_word2vec(loaded_compressed_text, loaded_model)
    print("Decompression successful, original text recovered.\n") if tokens == decompressed_text else print(
        "Decompression failed.\n")

    # Display Word2Vec-based comparison
    display_file_sizes(file_path, 'compressed_data_model.pkl')
    create_and_display_archives('original_data.txt', 'compressed_data_model.pkl')


def main(file_path: str) -> None:
    # Read the file in as pdf or regular text
    _, file_extension = os.path.splitext(file_path)
    tokens = read_pdf(file_path) if file_extension == '.pdf' else read_txt(file_path)

    # Frequency based approach
    frequency_based(tokens, file_path)

    # Word2Vec based approach
    word2vec(tokens, file_path)


if __name__ == "__main__":
    # fp = "../data/example_1.txt"
    # fp = "../data/example_2.txt"
    # fp = "../data/58924.pdf"
    fp = "../enwik9/enwik9"
    main(file_path=fp)
