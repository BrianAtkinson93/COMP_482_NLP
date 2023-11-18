import multiprocessing
import os
import re
import math
import sys
from collections import Counter

import logging
from multiprocessing import Pool

import torch as t
import pandas as pd
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from tqdm import tqdm

dir_ = os.path.abspath('articles')
file_list = os.listdir(dir_)

N = len(file_list)

vocabulary = []


def add_word(word):
    global vocabulary
    if not remove_non_letters(word) in vocabulary:
        vocabulary.append(remove_non_letters(word))


def extract_text(pdf_path, text):
    page_num = 0
    for page_layout in extract_pages(pdf_path):
        page_text = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text.append(element.get_text().strip())
        page_num += 1
        process_page(page_text, page_num, text)


def process_page(page_text, page_num, text):
    if len(page_text) > 0:
        part_num = 0
        for words in page_text[0:-1]:
            part_num += 1
            process_words(words, page_num, part_num, text)


def process_words(words, page_num, part_num, text):
    output = words.lower().replace('\n', ' ')
    text[-1].append(output)
    for word in words.split():
        add_word(word)


def getCountWordContext(text, w, c):
    dont_print = True
    logger.debug(f'w: {w}')
    logger.debug(f'c: {c}')
    count = 0

    # Not exact
    pattern = re.compile(r'' + w, re.I)
    context = re.compile(r'' + c, re.I)

    for doc in text:
        for chunk in doc:
            if context.search(chunk):
                j = 0
                match_i = -1
                words = chunk.split()
                length = len(words)
                for word in words:
                    if context.search(word):
                        match_i = j
                    j += 1

                if len(words) <= 0:
                    continue

                logger.debug('=====')
                logger.debug('found context ' + words[match_i] + ' matching ' + c + ' at index ' + str(match_i))
                for k in range(match_i - 4, match_i + 5):
                    if k == len(words) or abs(k) > len(words):
                        continue
                    test = words[k]
                    logger.debug('index = ' + str(k) + ', length of words ' + str(len(words)))
                    if 0 <= k < length:
                        if k == match_i:
                            logger.debug('-----')
                        logger.debug('checking ' + sanitize_string(words[k]) + ' for ' + w)

                        if pattern.search(test):
                            logger.debug('found (' + words[k] + ', ' + c + ')')
                            count += 1
    return count


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Check if either of the vectors is a zero-vector
    if norm_a == 0 or norm_b == 0:
        return 0

    dot_product = np.dot(a, b)
    return dot_product / (norm_a * norm_b)


def sanitize_string(s):
    return s.encode('ascii', 'ignore').decode('ascii')


def setup_logger(log_level):
    # Create a custom logger
    logger = logging.getLogger('ppmi')
    logging.basicConfig(level=logging.NOTSET, handlers=[])

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'ppmi.log')
    c_handler.setLevel(log_level)
    f_handler.setLevel(log_level)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def remove_non_letters(word):
    result = []
    for char in word:
        if char.isalpha():
            result.append(char)
    return ''.join(result)


# --- MAIN PROGRAM ---
if __name__ == "__main__":
    csv_file = 'output_table.csv'

    logger = setup_logger(20)

    logger.info('\n === Start of Program === \n')

    text = []
    for i in range(12):
        text.append([])
        path = dir_ + '\\' + str(i) + '.pdf'
        extract_text(path, text)

    logger.info(len(vocabulary))

    n = getCountWordContext(text, 'information', 'computer')
    logger.debug(f'n: {n}')

    vocab_words = ['community', 'students', 'indigenous', 'information', 'computer', 'analysis', 'student']
    context_words = ['research', 'children', 'project', 'computer', 'data', 'system', 'award']

    logger.info("Co-occurrence matrix")
    # Create co-occurrence matrix, populate with 0's
    co_occurrence_matrix = np.zeros((len(vocab_words), len(context_words)))

    logger.info('Populating Matrix')
    # populate the matrix with counts
    for i, word in tqdm(enumerate(vocab_words), total=len(vocab_words), desc="Processing"):
        for j, context in enumerate(context_words):
            logger.debug(f'i: {i}, word: {remove_non_letters(word)}')
            logger.debug(f'j: {j}, context: {remove_non_letters(context)}')

            co_occurrence_matrix[i][j] = getCountWordContext(text, remove_non_letters(word),
                                                             remove_non_letters(context))

    logger.info('Printing the matrix as a dataframe for viewing...')
    # Converting matrix to table with pandas
    df = pd.DataFrame(co_occurrence_matrix, index=vocab_words, columns=context_words)

    logger.info(f'writing to csv... {csv_file}')
    df.to_csv(csv_file)
    logger.info(df)

    total_counts = np.sum(co_occurrence_matrix)
    p_w = np.sum(co_occurrence_matrix, axis=1) / total_counts
    p_c = np.sum(co_occurrence_matrix, axis=0) / total_counts

    # Compute PPMI matrix from co_occurrence matrix
    # 1e-10 adds a small value so that we covert edge cases where logarithm could encounter a 0 value which is -infinity
    # maximum between value and 0 ensures that we have positive values
    logger.info("Computing PPMI matrix")
    PPMI = np.maximum(np.log2((co_occurrence_matrix / (p_w[:, None] * p_c[None, :] + 1e-10)) + 1e-10), 0)

    # compute cosine similarity between rows
    num_words1 = len(vocab_words)
    num_words2 = len(context_words)
    similarities = np.zeros((num_words1, num_words2))

    logger.info("Finding cosine_similarities")
    # The cosine_similarity function is based on the textbook pseudocode (page. 113)
    for i in range(num_words1):
        for j in range(num_words2):
            similarities[i][j] = cosine_similarity(PPMI[i], PPMI[j])

    logger.info(f'\nCosine Similarity Matrix:')
    for i, word1 in enumerate(vocab_words):
        for j, word2 in enumerate(context_words):
            logger.info(f'Cosine similarity between {word1} and {word2}: {similarities[i][j]}')
