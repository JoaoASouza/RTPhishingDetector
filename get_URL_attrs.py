from datahelper import *
from datetime import datetime
import time

domains_data = []

LEGITIMATE_DATA = True
USE_EXTERN_TOOLS = False

if (LEGITIMATE_DATA):
    FILE_NAME = "legitimate_urls_moz_" + datetime.today().strftime('%Y-%m-%d')
    OUT_FILE_NAME = "out" + datetime.today().strftime('%Y-%m-%d') + "_legitimate.csv"
else:
    FILE_NAME = "phishing_urls_phishtank_" + datetime.today().strftime('%Y-%m-%d')
    OUT_FILE_NAME = "out" + datetime.today().strftime('%Y-%m-%d') + ".csv"

time_sum = 0
time_count = 0

try:
    with open("lists/" + FILE_NAME) as urls_file:
        urls = urls_file.readlines()
        for url in urls:
            url = url[:-1]

            start_time = time.time()

            domain = urlparse(url).netloc
            protocol = urlparse(url).scheme

            print("\n\n")
            print(url)

            print("checking...")
            response = check_url(url)
            if (response == 0 or not response):
                continue

            domain_data = {}

            domain_data['url'] = url
            domain_data['url_length'] = len(url)
            print("is valid ip")
            domain_data['is_ip'] = is_valid_ip(domain)
            domain_data['url_has_at_char'] = '@' in url
            domain_data['number_of_dots'] = url.count('.')
            domain_data['is_https'] = protocol == 'https'
            if (USE_EXTERN_TOOLS):
                print("get domain age")
                domain_data['domain_age'] = get_domain_age(domain)
                print("get domain page rank")
                domain_data['page_rank'] = get_domain_page_rank(domain)
            domain_data['url_redirection'] = url.count("//") > 1
            domain_data['domain_has_dash_char'] = '-' in domain
            domain_data['url_has_https_token'] = False if 'https' not in url else url.index('https') != 0

            if (USE_EXTERN_TOOLS):
                print('get DNS data')
                dns_data = get_dns_data(domain)
                domain_data.update(dns_data)

            print('get HTML data')
            html_data = get_html_data(url, response)
            domain_data.update(html_data)

            elapsed = time.time() - start_time
            time_sum += elapsed
            time_count += 1
            print("ELAPSED TIME =", elapsed)

            domains_data.append(domain_data)

            # print(domains_data)

            print("writing...")
            write_data_to_file(domains_data, "datasets/" + OUT_FILE_NAME)
            print("writing done!")

finally:
    print("\n\n>>>AVERAGE TIME =", time_sum/time_count)
