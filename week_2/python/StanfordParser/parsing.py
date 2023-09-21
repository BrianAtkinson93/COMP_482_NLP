import os
import re
import codecs
import csv
import pandas as pd

from PyMultiDictionary import MultiDictionary, DICT_WORDNET
from nltk.parse.corenlp import CoreNLPParser
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


# https://stanfordnlp.github.io/CoreNLP/download.html
# Navigate to the Stanford CoreNLP install folder and run the CoreNLPServer
#   so, e.g.: folder C:\Folders\Projects\Python\Stanford\stanford-corenlp-4.5.4
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# CTRL-C shuts the server down fine

def add_record(word, d):
    # print(word)
    global dictionaries
    t = dictionaries[d].get(word)
    if t is not None:
        t += 1
        dictionaries[d].update({word: t})
    else:
        dictionaries[d].update({word: 1})
    global vocabulary
    v = vocabulary.get(word)
    if v is not None:
        v += 1
        vocabulary.update({word: v})
    else:
        vocabulary.update({word: 1})


def extract_text(pdf_path, d):
    for page_layout in extract_pages(pdf_path):
        page_text = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text.append(element.get_text().strip())
        # Remove header and footer lines
        for chunk in page_text:
            if len(chunk) > 0:
                parse = next(parser.raw_parse(chunk))
            for item in parse.leaves():
                if 'http' not in item:
                    item = item.lower()
                    add_record(item, d)


if __name__ == "__main__":

    parser = CoreNLPParser(url='http://localhost:9000')
    pdfdirectory = 'C:\\Users\\brian\\Desktop\\School\\Fall_2023\\COMP_482_NLP\\week_2\\data\\'
    directory = os.listdir(pdfdirectory)

    directory = [file for file in directory if file.endswith('.pdf')]  # Added list comprehension to exclude none .pdfs

    N = len(directory)

    # replacement for collection
    dictionaries = []
    for i in range(0, N):
        dictionaries.append({})

    vocabulary = {}

    print()
    print('--- PROCESSING DOCUMENTS ---')
    print()

    temp = codecs.open('temp.csv', 'w', "utf-8")
    temp_writer = csv.writer(temp)

    for i in range(0, N):
        print('Processing file ' + str(i) + ': ' + directory[i])
        extract_text(pdfdirectory + directory[i], i)
        col = [directory[i]]
        row = [directory[i]]
        for r in sorted(dictionaries[i]):
            # print(str(dictionaries[i][r]) + ' ' + r)  # Commented out to increase speed as IO is expensive
            row.append(dictionaries[i][r])
            col.append(str(r))
        temp_writer.writerow(col)
        temp_writer.writerow(row)

    print()
    print('--- ALL VOCABULARY ---')
    print()

    f1 = codecs.open('freq_counts.csv', 'w', "utf-8")
    full_writer = csv.writer(f1)

    col_titles = ['']
    for r in sorted(vocabulary):
        col_titles.append(str(r))
    full_writer.writerow(col_titles)

    count = 0
    for dic in dictionaries:
        row = []
        row.append(directory[count])
        for r in sorted(vocabulary):
            freq = dic.get(r)
            if freq is not None:
                row.append(freq)
            else:
                row.append(0)
            # print(str(vocabulary[r]) + ' ' + r)
        full_writer.writerow(row)
        count += 1

    total = ['total']

    for r in sorted(vocabulary):
        total.append(vocabulary[r])
    full_writer.writerow(total)
