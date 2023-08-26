import requests
from bs4 import BeautifulSoup
from datetime import datetime

FILE_NAME = "phishing_urls_phishtank_" + datetime.today().strftime('%Y-%m-%d')

def get_url_in_detail_page(phish_id):
    url = "https://phishtank.org/phish_detail.php?phish_id=" + phish_id

    response = requests.get(url)

    if response.ok:
        soup = BeautifulSoup(response.text, "html.parser")
        bold_elements = soup.find_all('b')
        print("BOLD ELEMENTS")
        print(bold_elements)
        for element in bold_elements:
            if not element.find('a'):
                return element.text


def get_open_phish_urls():

    url = "https://openphish.com/feed.txt"

    response = requests.get(url)

    if response.ok:
        urls = response.text
        return [url + "\n" for url in urls.split("\n")[:-1]]


urls = get_open_phish_urls()
# urls = []

page_num = 0
while True:

    print("PAGE = ", page_num)

    url = "https://phishtank.org/phish_search.php?page={}&active=y&verified=y".format(page_num)

    response = requests.get(url)

    if response.ok:
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find('table')
        rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all('td')
            if len(cells) == 0:
                continue

            cells_text = [cell.text for cell in cells]

            phish_id = cells_text[0]
            url = cells_text[1]

            is_url_complete = False if url.find("...") != -1 else True

            if is_url_complete:
                url = url.split("added")[0] + "\n"
            else:
                print("INCOMPLETE")
                print(url)
                full_url = get_url_in_detail_page(phish_id)
                if not full_url:
                    continue
                url = full_url + "\n"

            if url not in urls:
                urls.append(url)

            with open("lists/" + FILE_NAME, "w+") as file:
                file.writelines(urls)

    page_num += 1