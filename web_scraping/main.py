import csv
import sys

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


def safe_xpath_string_literal(input_str):
    """ creates a concatenation of alternately-quoted strings that is always a valid XPath expression """
    parts = input_str.split("'")
    return "concat('" + "', \"'\" , '".join(parts) + "', '')"


def safe_xpath_string_literal_2(string):
    print(f'    Original: {string}')
    if '"' in string:
        parts = string.split('"')
        new = "concat('" + "', '\"', '".join(parts) + "', '')"
        print(f'    New1: {new}')
        return new
    elif "'" in string:
        parts = string.split("'")
        new = 'concat("' + '", "\'", "'.join(parts) + '", "")'
        print(f'    New2: {new}')
        return new
    return f"'{string}'"


# Initial Setup
print('Starting..')
driver = webdriver.Chrome()
starting_url = "https://www.gigabyte.com/Support/FAQ"
driver.get(starting_url)
WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.spanLink")))
question_texts = [q.text for q in driver.find_elements(By.CSS_SELECTOR, "span.spanLink")]
with open('FAQ_Scrapes.txt', 'w', encoding='utf-8') as f, open("FAQ_Scrapes.csv", 'w', newline='', encoding='utf-8') as csvf:
    csvwriter = csv.writer(csvf)
    csvwriter.writerow(['Number', 'Question', 'Answer'])
    num = 1
    f.write(f'Questions and Answers from {starting_url}\n\n')
    f.flush()
    try:
        while True:
            print(f"The current URL is: {driver.current_url}")
            WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.spanLink")))

            question_texts = [q.text for q in driver.find_elements(By.CSS_SELECTOR, "span.spanLink")]
            # question_texts = [q.text for q in driver.find_elements(By.CSS_SELECTOR, "span.spanLink")]

            for question_text in question_texts:
                try:
                    xpath_expression = f"//span[contains(text(), {safe_xpath_string_literal(question_text)})]"
                    question = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, xpath_expression)))

                    # Scroll into view
                    driver.execute_script("arguments[0].scrollIntoView(true);", question)
                    time.sleep(1)  # Adjust sleep time as needed
                    # Scroll into view with additional space
                    driver.execute_script("window.scrollBy(0, -150);", question)  # Scrolls up a bit to avoid header

                    try:
                        question.click()
                    except Exception as click_exception:
                        print("Normal click failed, trying JavaScript click...")
                        driver.execute_script("arguments[0].click();", question)

                    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, "FaqSolution")))
                    answer = driver.find_element(By.CLASS_NAME, "FaqSolution").text

                    print(f"Question: {question_text}")
                    print(f"Answer: {answer}\n")

                    f.write("Q - " + question_text + "\n")
                    f.write("A - " + answer + "\n\n")
                    f.flush()
                    csvwriter.writerow([num, question_text, answer])
                    csvf.flush()
                    num += 1

                    driver.back()

                except TimeoutException as e:
                    print(f'Timeout occurred with question: {question_text}')
                    # print(f'Error: {e}')
                    driver.back()
                    continue
                except NoSuchElementException as e:
                    print(f'Element not found for question: {question_text}')
                    # print(f'Error: {e}')
                    driver.back()
                    continue
                except Exception as e:
                    print(f'Unexpected error with question: {question_text}')
                    # print(f'Error: {e}')
                    driver.back()
                    continue
            try:
                print("Looking for next button...")
                next_button = WebDriverWait(driver, 30).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "btn-group-next"))
                )

                # Scroll the next button into view
                driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                time.sleep(2)  # Wait for any overlays to disappear

                try:
                    next_button.click()
                    print('Clicked Button...')
                    time.sleep(5)
                except Exception as click_exception:
                    print("Normal click failed, trying JavaScript click...")
                    driver.execute_script("arguments[0].click();", next_button)

            except NoSuchElementException:
                print("No 'Next' button found.")
                # Handle the absence of the next button if necessary

    finally:
        driver.quit()
