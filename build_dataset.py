import glob
from urllib.parse import urlparse

data_files = glob.glob("datasets/out*.csv")
legitimate_data_files = glob.glob("datasets/out*_legitimate.csv")

phishing_data_files = [item for item in data_files if item not in legitimate_data_files]

phishing_domains = []
phishing_data = []

with open("datasets/dataset.csv", "w+") as out_file:
    out_file.write("url,url_length,is_ip,url_has_at_char,number_of_dots,is_https,domain_age,page_rank,url_redirection,domain_has_dash_char,url_has_https_token,has_DNS_record,A_records,avg_ttl,asn_count,AAAA_records,avg_AAAA_ttl,has_iframe,most_frequent_domain_in_anchor,most_frequent_domain_in_link,most_frequent_domain_in_source,common_page_rate,footer_common_page_rate,has_anchor_in_body,null_link_ratio,footer_null_link_ratio,phishing\n")

for phishing_data_file in phishing_data_files:
    with open(phishing_data_file, "r") as file:

        data = file.read().split("\n")

        del data[0]
        del data[-1]

        for row in data:
            url = row.split(",")[0]
            domain = urlparse(url).netloc

            if (domain in phishing_domains):
                continue

            phishing_domains.append(domain)
            phishing_data.append(row + "True\n")

with open("datasets/dataset.csv", "a") as out_file:
    out_file.writelines(phishing_data)

legitimate_domains = []
legitimate_data = []

for legitimate_data_file in legitimate_data_files:
    with open(legitimate_data_file, "r") as file:

        data = file.read().split("\n")

        del data[0]
        del data[-1]

        for row in data:
            url = row.split(",")[0]
            domain = urlparse(url).netloc

            if (domain in legitimate_domains):
                continue

            legitimate_domains.append(domain)
            legitimate_data.append(row + "False\n")

with open("datasets/dataset.csv", "a") as out_file:
    out_file.writelines(legitimate_data)