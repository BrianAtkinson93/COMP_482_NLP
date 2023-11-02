import os
import pickle
import zipfile
import tarfile
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import PyPDF2


def get_file_size(file_path, units='MB'):
    size_in_bytes = os.path.getsize(file_path)
    if units == 'KB':
        return size_in_bytes / 1024
    elif units == 'MB':
        return (size_in_bytes / 1024) / 1024
    elif units == 'GB':
        return ((size_in_bytes / 1024) / 1024) / 1024
    else:
        return size_in_bytes


def read_pdf(file_path):
    tokens = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)
        for i in tqdm(range(total_pages), desc="Reading PDF pages", unit="page"):
            page = pdf_reader.pages[i]
            text = page.extract_text()
            tokens.extend(word_tokenize(text.lower()))
    return tokens


def read_txt(file_path):
    tokens = []
    total_lines = sum(1 for line in open(file_path, 'r', encoding='utf-8'))
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Reading lines", unit="line"):
            try:
                tokens.extend(word_tokenize(line.lower()))
            except UnicodeDecodeError:
                continue
    return tokens


def compress_text(tokens):
    model = Word2Vec([tokens], vector_size=50, window=5, min_count=1, sg=0)
    compressed_text = [model.wv.index_to_key.index(word) for word in tokens if word in model.wv.index_to_key]
    return compressed_text, model


def decompress_text(compressed_text, model):
    return ' '.join([model.wv.index_to_key[index] for index in compressed_text])


def main(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.pdf':
        tokens = read_pdf(file_path)
    else:
        tokens = read_txt(file_path)

    compressed_text, model = compress_text(tokens)
    decompressed_text = decompress_text(compressed_text, model)

    # Save and load compressed data
    with open('compressed_data.pkl', 'wb') as f:
        pickle.dump(compressed_text, f)
    with open('compressed_data.pkl', 'rb') as f:
        loaded_compressed_text = pickle.load(f)

    # Verify if loaded compressed data is identical to original compressed data
    if compressed_text == loaded_compressed_text:
        print("Verification successful, loaded compressed data is identical to original.")
    else:
        print("Verification failed, loaded compressed data is not identical to original.")

    # Verify decompression
    original_text = ' '.join(tokens)
    if original_text == decompressed_text:
        print("Decompression successful, original text recovered.\n")
    else:
        print("Decompression failed.\n")

    # Display file sizes
    original_size = get_file_size(file_path)
    compressed_size = get_file_size('compressed_data.pkl')
    print(f"Size of original file: {original_size} MB")
    print(f"Size of my compressed file: {compressed_size} MB")

    # # Create ZIP and tar.gz files without overwriting the original
    # with zipfile.ZipFile('compressed_data.zip', 'w') as zipf:
    #     zipf.write('compressed_data.pkl')
    # with tarfile.open('compressed_data.tar.gz', 'w:gz') as tar:
    #     tar.add('compressed_data.pkl')
    #
    # # Display ZIP and tar.gz sizes
    # zip_size = get_file_size('compressed_data.zip')
    # tar_size = get_file_size('compressed_data.tar.gz')
    # print(f"Size of ZIP file: {zip_size} MB, diff: {zip_size - compressed_size}")
    # print(f"Size of tar.gz file: {tar_size} MB, diff: {tar_size - compressed_size}")

    print(f'\n')

    # Save original text to a txt file
    with open('original_data.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(tokens))

    # Create ZIP and tar.gz files for original text
    with zipfile.ZipFile('original_data.zip', 'w') as zipf:
        zipf.write('original_data.txt')
    with tarfile.open('original_data.tar.gz', 'w:gz') as tar:
        tar.add('original_data.txt')

    # Display ZIP and tar.gz sizes for original text
    zip_size_original = get_file_size('original_data.zip')
    tar_size_original = get_file_size('original_data.tar.gz')
    print(f"Size of original ZIP file: {zip_size_original} MB, diff: {compressed_size - zip_size_original}")
    print(f"Size of original tar.gz file: {tar_size_original} MB, diff: {compressed_size - tar_size_original}")


if __name__ == "__main__":
    # fp = "../data/example_1.txt"
    # fp = "../data/example_2.txt"
    fp = "../data/58924.pdf"
    main(file_path=fp)
