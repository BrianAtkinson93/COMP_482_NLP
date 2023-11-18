# Gigabyte FAQ Scraper

---

## Overview
This script is designed to scrape FAQ questions and answers from the Gigabyte website. It navigates through the FAQ sections, extracts each question and its corresponding answer, and saves them in both text and CSV formats.

## Features
- Scrape FAQ questions and answers from Gigabyte's official website.
- Save the scraped data into a `.txt` file and a `.csv` file.
- Handle pagination to scrape FAQs from multiple pages.
- Robust error handling to manage timeouts and missing elements.

## Prerequisites
- Python 3.9.13
- selenium==4.15.2
- Chrome WebDriver

## Usage
1. Ensure Python 3 and Selenium are installed.
2. Download the Chrome WebDriver and place it in an accessible location.
3. Update the script to point to the WebDriver's location.
4. Run the script using `python3 main.py` or `python.exe main.py`.
5. Check the `FAQ_Scrapes.txt` and `FAQ_Scrapes.csv` files for the output.

## Function Descriptions
- `safe_xpath_string_literal(input_str)`: Handles XPath string literals to avoid errors due to quotes.
- `safe_xpath_string_literal_2(string)`: Another method to handle XPath string literals, covering different scenarios.

## Output Format
- `FAQ_Scrapes.txt`: Plain text format containing questions and answers.
- `FAQ_Scrapes.csv`: CSV format with columns 'Number', 'Question', 'Answer'.

## Error Handling
The script includes error handling for:
- Timeout Exceptions: When a page or element takes too long to load.
- No Such Element Exceptions: When an expected element is not found on the page.
- General Exceptions: Covers any other unexpected issues.

## Note
This script is only intended for educational purposes and should be used responsibly. Ensure compliance with Gigabyte's website terms of use and scraping policies.

## Author
* Brian Atkinson  

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### MIT License Summary
- Allows commercial use, modification, distribution, private use
- Provides an express grant of patent rights from contributors
- The software is provided "as is", without warranty of any kind

### Full License Text
Copyright 2023 Brian Atkinson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Usage of External Libraries
If your script uses external libraries, make sure to comply with their respective licenses. For example, Selenium is licensed under the Apache License 2.0. 

### Disclaimer
This script is intended for educational purposes only. Users are responsible for ensuring they comply with Gigabyte's terms of service regarding web scraping.


## Selenium License
This script uses Selenium, which is an open-source project. Selenium is licensed under the [Apache License 2.0](https://www.selenium.dev/documentation/about/copyright/).

### Apache License 2.0 Summary
- Permits commercial use, modification, distribution, and private use.
- Provides an express grant of patent rights from contributors.
- Comes with a trademark disclaimer.
- The software is provided "as is", without warranty of any kind.

For more details on the Selenium license, visit their [copyright page](https://www.selenium.dev/documentation/about/copyright/).

