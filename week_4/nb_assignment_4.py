"""
Brian Atkinson | 300088157
Monday, October 2nd 2023 | Fall 2023
COMP - 482 | Natural Language Processing
Dr. Russell Campbell
University of the Fraser Valley

Assignment 2

Use your PDF documents from Assignment 1 and decide one of the documents as a “positive” class and a second document as
a “negative” class. Keep a third document for testing.

Write a program in Python to calculate the predicted class of the third document based on a naïve Bayes text
classification model with Laplace smoothing. You should have the counts in a csv from the first assignment to lookup
the word counts in each document.

Submit zipped folder with code and by Tuesday, Oct 3, before 5 pm.

Rubric 10 marks total
• [6 marks] correct formula calculations for training
• [1 mark] output statements to show calculations for top 10 most frequent words
• [1 mark] comments to explain your code
• [1 mark] log to avoid underflow
• [1 mark] classification of an input document output to stdout.
"""

import argparse
import pandas as pd
import math
import sys

from typing import Tuple, List, Dict
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def read_in_csv(csv_file: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Read the CSV file and return a DataFrame after filtering out unwanted columns.

    :param csv_file: String representation of the file to read in.
    :return: Tuple of the DataFrame, and List of Titles
    """
    print(f'Reading in {csv_file}...')
    dataframe = pd.read_csv(csv_file, encoding='utf-8')

    # Extract the pdf names from the DataFrame
    titles = dataframe.iloc[:, 0].to_list()

    # Filter out columns with special characters, numbers, etc.
    cleaned_column_names = [col for col in dataframe.columns if col.isalpha() and col.isascii()]

    # Filter the DataFrame based on cleaned column names
    filtered_dataframe = dataframe[cleaned_column_names]

    return filtered_dataframe, titles


def naive_bayes(test_doc: Dict[str, int], train_doc: Dict[str, int], total_train: int, unique_words: int,
                type_: str, log_prior: float) -> float:
    """
    Calculate the Naive Bayes log probability for a given test document based on a training document.

    :param test_doc: A dictionary representing the test document. The keys are words and the values are their
                        frequencies in the test document.
    :param train_doc: A dictionary representing the training document. The keys are words and the values are their
                        frequencies in the training document.
    :param total_train: The total number of words in the training document.
    :param unique_words: The number of unique words across both the training and test documents.
    :param type_: The type of PDF being analyzed (e.g., 'Good', 'Bad', etc.).
    :param log_prior:
    :return float: The calculated Naive Bayes log probability for the test document based on the training document.

    Example:
    >>> naive_bayes({'apple': 1, 'banana': 1}, {'apple': 1, 'banana': 2, 'cherry': 1}, 4, 3, 'Good')
    -2.0794415416798357
    """
    print(f'Calculating Naive Bayes probabilities for {type_} pdf...')
    log_prob = 0.0
    for word, count in test_doc.items():
        # Calculate the frequency of the word in the training document
        freq = train_doc.get(word, 0)

        # Calculate the probability of the word
        prob_word = (freq + 1) / (total_train + unique_words)

        # Update the log probability
        log_prob += count * math.log(prob_word)

    # Add the log_prior to the final log_prob
    log_prob += log_prior

    return log_prob


def generate_pdf_report(classification_result: str, top_words: List[str], document_titles: List[str]) -> None:
    """
    Generate a PDF report that includes the classification result, top words, and document titles.

    :param classification_result: The result of the document classification, usually a string indicating the class or category.
    :param top_words: A list of strings representing the top words to be included in the report.
    :param document_titles: A list of strings representing the titles of the documents being analyzed.

    :return: None. The function saves the report as a PDF file.
    """
    pdf = SimpleDocTemplate(
        "Classification_Report.pdf",
        pagesize=letter
    )

    # Container for the 'Flowable' objects
    elements = []

    # Add title and classification result
    styles = getSampleStyleSheet()
    elements.append(Paragraph('Classification Report', styles['Title']))
    elements.append(Spacer(1, 12))
    document_title_style = styles['Normal'].clone('Normal')
    document_title_style.fontSize = 10
    document_title_style.fontName = 'Helvetica-BoldOblique'  # Bold and Italic
    document_title_style.alignment = TA_LEFT
    elements.append(Paragraph(f"Good pdf: {document_titles[0]}", document_title_style))
    elements.append(Paragraph(f"Bad pdf: {document_titles[1]}", document_title_style))
    elements.append(Paragraph(f"Tested pdf: {document_titles[2]}", document_title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f'Classification Result: {classification_result}', styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add table for top words
    data = [["Word", "Count"]] + top_words
    t = Table(data)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                           ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                           ('FONTSIZE', (0, 0), (-1, 0), 14),
                           ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                           ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                           ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(t)

    # Generate PDF
    pdf.build(elements)


def main(args):
    """ Main Flow of the program... """
    # Read CSV.
    dataframe, titles = read_in_csv(args.input_csv)

    # Extract rows for good_pdf and bad_pdf.
    good_pdf_row = dataframe.iloc[args.good_pdf]
    bad_pdf_row = dataframe.iloc[args.bad_pdf]
    test_pdf_row = dataframe.iloc[args.test_pdf]

    # Convert DataFrames to dicts.
    positive_doc = good_pdf_row.to_dict()
    negative_doc = bad_pdf_row.to_dict()
    test_doc = test_pdf_row.to_dict()

    # Get Titles of pdfs in question.
    titles = [titles[args.good_pdf], titles[args.bad_pdf], titles[args.test_pdf]]

    # Calculate total word counts for each document.
    total_positive = sum({k: v for k, v in positive_doc.items()}.values())
    total_negative = sum({k: v for k, v in negative_doc.items()}.values())

    # Calculate unique words in the corpus.
    # Set only stores single instances, and is highly efficient.
    unique_words = len(set(list(positive_doc.keys()) + list(negative_doc.keys()) + list(test_doc.keys())))

    # Calculate the prior probabilities based on your training data
    total_pdfs = 2
    num_good_pdfs = 1
    num_bad_pdfs = 1

    prior_good = num_good_pdfs / total_pdfs
    prior_bad = num_bad_pdfs / total_pdfs

    # Convert to log scale
    log_prior_good = math.log(prior_good)
    log_prior_bad = math.log(prior_bad)

    # Calculate Naive Bayes probabilities for the test document
    prob_positive_given_test = naive_bayes(test_doc, positive_doc, total_positive, unique_words, 'Positive',
                                           log_prior_good)
    prob_negative_given_test = naive_bayes(test_doc, negative_doc, total_negative, unique_words, 'Negative',
                                           log_prior_bad)


    """
    Top 10 most frequent words in the test document and their calculations.
    
    Sorting based on the key: specifying the value to sort by.
    Defining that we want the list to be sorted in reverse order (largest to smallest)
    Then we slice the list to only extract the top 10 items ( allowing for modularity in the future)
    Example: {'apple': 1, 'banana': 3, 'orange' : 2}
    Lambda x: x[1] will pick the value not the key, and sort based on this, you can refine the lambda function
    to classify in any means you feel appropriate.
    output: [('banana' : 3), ('orange' : 2), ('apple', 1)]
    """
    sorted_test = sorted(test_doc.items(), key=lambda x: x[1], reverse=True)[:args.top_n]

    # Log to avoid underflow.
    # Underflow happens when numbers near zero and rounded to zero, causing loss of information.
    prob_positive_given_test = math.exp(prob_positive_given_test)
    prob_negative_given_test = math.exp(prob_negative_given_test)

    # Classification of document to stdout.
    if prob_positive_given_test > prob_negative_given_test:
        classification_result = "The test document is classified as belonging to the positive class."
        print(classification_result)
    else:
        classification_result = "The test document is classified as belonging to the negative class."
        print(classification_result)

    # Generate a pdf report for the classification
    generate_pdf_report(classification_result, sorted_test, titles)

    # Return code of 0 to define successful execution
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_csv', type=str, help="Please define the frequency count csv you'd like to use.")
    parser.add_argument('--good_pdf', type=int, help="Please specify the row that you want to define as good.")
    parser.add_argument('--bad_pdf', type=int, help="Please specify the row that you want to define as bad.")
    parser.add_argument('--test_pdf', type=int, help="Please define the row that you want to classify")
    parser.add_argument('--output_name', type=str, default="classification.pdf",
                        help="Please specify the name of the output csv if you'd like to use something other than default")
    parser.add_argument('--top_n', type=int, default=10,
                        help="Please provide n to define the # of top values to display in report.")

    arguments = parser.parse_args()

    sys.exit(main(arguments))
