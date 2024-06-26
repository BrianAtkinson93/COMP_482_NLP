# Create PDF from Webpage in Python
# Nikita Tonkoshkur
# 2021 Aug 4
# last accessed: 2023 Jun 6
# https://medium.com/@nikitatonkoshkur25/create-pdf-from-webpage-in-python-1e9603d6a430

# Adapted program by Russell Campbell to check access to one webpage works.

import os
import sys

from requests_html import HTMLSession
import base64
import json
import time
from io import BytesIO
from typing import List

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class PdfGenerator:
    """
     Simple use case:
        pdf_file = PdfGenerator(['https://google.com']).main()
        with open('new_pdf.pdf', "wb") as outfile:
            outfile.write(pdf_file[0].getbuffer())
    """
    driver = None
    # Google documentation
    # https://chromedevtools.github.io/devtools-protocol/tot/Page#method-printToPDF
    print_options = {
        'landscape': False,
        'displayHeaderFooter': True,
        'printBackground': False,
        'preferCSSPageSize': True,
        'paperWidth': 8.5,
        'paperHeight': 11,
    }

    def __init__(self, urls: List[str]):
        self.urls = urls

    def _get_pdf_from_url(self, url, *args, **kwargs):
        self.driver.get(url)
        try:
            element_present = EC.presence_of_element_located((By.ID, 'Id_of_element_to_be_present'))
            WebDriverWait(self.driver, 10).until(element_present)
        except TimeoutException:
            print("Timed out waiting for page to load")

        print_options = self.print_options.copy()
        result = self._send_devtools(self.driver, "Page.printToPDF", print_options)
        return base64.b64decode(result['data'])

    # def _get_pdf_from_url(self, url, *args, **kwargs):
    #     self.driver.get(url)
    #
    #     # this is a bit hacky --- maybe someone can figure out a way to wait for necessary time only
    #     time.sleep(5)  # allow the page to load, increase if needed
    #
    #     print_options = self.print_options.copy()
    #     result = self._send_devtools(self.driver, "Page.printToPDF", print_options)
    #     return base64.b64decode(result['data'])

    @staticmethod
    def _send_devtools(driver, cmd, params):
        """
        Works only with chromedriver.
        Method uses cromedriver's api to pass various commands to it.
        """
        resource = "/session/%s/chromium/send_command_and_get_result" % driver.session_id
        url = driver.command_executor._url + resource
        body = json.dumps({'cmd': cmd, 'params': params})
        response = driver.command_executor._request('POST', url, body)
        return response.get('value')

    def _generate_pdfs(self):
        pdf_files = []

        for url in self.urls:
            result = self._get_pdf_from_url(url)
            file = BytesIO()
            file.write(result)
            pdf_files.append(file)

        return pdf_files

    def _convert2pdf(self, html, filename):
        page_file = open('temp.html', 'w')
        page_file.write(html)
        page_file.close()
        self.driver.get(os.getcwd() + '/temp.html')
        print_options = self.print_options.copy()
        resource = "/session/%s/chromium/send_command_and_get_result" % self.driver.session_id
        url = self.driver.command_executor._url + resource
        body = json.dumps({'cmd': "Page.printToPDF", 'params': print_options})
        response = self.driver.command_executor._request('POST', url, body)
        result = response.get('value')
        data = base64.b64decode(result['data'])
        file = BytesIO()
        file.write(data)
        with open(filename, "wb") as outfile:
            outfile.write(file.getbuffer())
        print('done writing pdf')

    def main(self, html, filename) -> List[BytesIO]:
        webdriver_options = ChromeOptions()
        webdriver_options.add_argument('--headless')
        webdriver_options.add_argument('--disable-gpu')

        try:
            self.driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()),
                options=webdriver_options
            )
            # result = self._generate_pdfs()
            result = self._convert2pdf(html, filename)
        finally:
            self.driver.close()
        print('done with main conversion')

        return result


if __name__ == "__main__":
    # MAIN PROGRAM

    session = HTMLSession()

    urls = [
        "https://www.brookings.edu/articles/ukraine-and-the-kinzhal-dont-believe-the-hypersonic-hype",
        "https://www.cbo.gov/publication/58924",
        "https://theconversation.com/how-hypersonic-missiles-work-and-the-unique-threats-they-pose-an-aerospace-engineer-explains-180836",
        "https://www.armscontrol.org/act/2023-03/news/us-faces-wins-losses-hypersonic-weapons",
        ""
    ]

    main_url = "https://www.cbo.gov/publication/58924"
    file_name = os.path.basename(main_url)
    save_path = f'../../data/{file_name}.pdf'
    print(f'Creating file {save_path}')

    r = session.get(main_url)
    "byo-block -narrow wysiwyg-block wysiwyg"
    contents = r.html.find('.At-A-Glance-Text-Frame', _encoding='utf-8')

    PdfGenerator([]).main(contents[0].html, save_path)
