import requests
from bs4 import BeautifulSoup
from datetime import datetime

FILE_NAME = "legitimate_urls_moz_" + datetime.today().strftime('%Y-%m-%d')

urls = []

with open("moztop500.html", "r") as html_file:

    html_content = html_file.read()

    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all('table')

    urls = []

    for table in tables:

        table_body = table.find('tbody')
        table_rows = table_body.find_all('tr')

        for row in table_rows:
            anchor = row.find('a', {"class": "ml-2"})
            urls.append(anchor['href'] + "\n")

with open("lists/" + FILE_NAME, "w+") as file:
    file.writelines(urls)