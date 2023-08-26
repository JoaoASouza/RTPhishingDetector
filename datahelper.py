import requests
from IPy import IP
import dns
import dns.resolver
import socket, struct
import whois
import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def check_url(url):
    try:
        response = requests.get(url, timeout=30)
        if (response.ok):
            return response
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.TooManyRedirects) as e:
        return 0

def check_domain(domain, prefix):
    try:
        response = requests.get(prefix + "://" + domain, timeout=30)
        if (response.ok):
            return response
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.TooManyRedirects) as e:
        return 0

def is_valid_ip(addr):
    try:
        IP(addr)
        return True
    except ValueError:
        return False

def write_data_to_file(domains_data, file_name):
    with open(file_name, "w+") as file:
        
        for key in domains_data[0]:
            file.write(key)
            file.write(",")
        file.write("\n")

        for i in range(0, len(domains_data)):
            for key in domains_data[i]:
                # print(domains_data[i][key])
                file.write(str(domains_data[i][key]))
                file.write(",")
            file.write("\n")

def find_asn(ip, asn_data):
    ip_num = ip_to_number(ip)
    for data in asn_data:
        if (data['range_start'] <= ip_num <= data['range_end'] ):
            return data['AS_number']

def get_asn_count(records, asn_data):
    asns = []
    for record in records:
        ip = record.split(" ")[1]
        asns.append(find_asn(ip, asn_data))
    return len([*set(asns)])

def get_dns_data(domain):

    asn_data = load_asn_data("./ip2asn-v4.tsv")

    domain_data = {}

    try:
        dns_result = dns.resolver.resolve(domain)
        records = dns_result.rrset.to_text().split("\n")

        ttls = []
        for record in records:
            ttls.append( int(record.split(" ")[1]) )

        print("\thas_DNS_record")
        domain_data["has_DNS_record"] = True
        print("\tA_records")
        domain_data["A_records"] = len(records)
        print("\tavg_ttl")
        domain_data["avg_ttl"] = sum(ttls) / len(ttls)
        print("\tasn_count")
        domain_data["asn_count"] = get_asn_count(records, asn_data)
    except dns.resolver.NXDOMAIN:
        domain_data["has_DNS_record"] = False
        domain_data["A_records"] = 0
        domain_data["avg_ttl"] = 0
        domain_data["asn_count"] = 0

    try:
        dns_aaaa_result = dns.resolver.resolve(domain, "AAAA")
        aaaa_records = dns_aaaa_result.rrset.to_text().split("\n")

        aaaa_ttls = []
        for record in aaaa_records:
            aaaa_ttls.append( int(record.split(" ")[1]) )

        print("\tAAAA_records")
        domain_data["AAAA_records"] = len(aaaa_records)
        print("\tavg_AAAA_ttl")
        domain_data["avg_AAAA_ttl"] = sum(aaaa_ttls) / len(aaaa_ttls)
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
        domain_data["AAAA_records"] = 0
        domain_data["avg_AAAA_ttl"] = 0

    return domain_data

def is_domain_more_frequent_in_tag(soup, current_domain, tag):

    found_tags = soup.find_all(tag, attrs={"href": True})

    if (len(found_tags) == 0):
        return True

    tag_urls = [t['href'] for t in found_tags]

    domain_freq_dict = {}
    for u in tag_urls:
        domain = urlparse(u).netloc
        if (not domain):
            domain = current_domain
        if (not domain in domain_freq_dict):
            domain_freq_dict[domain] = 0
        domain_freq_dict[domain] += 1

    most_frequent_domain = max(domain_freq_dict, key=domain_freq_dict.get)
    return most_frequent_domain == current_domain

def get_page_detection_rate(soup):

    found_anchors = soup.find_all('a', attrs={"href": True})

    if (len(found_anchors) == 0):
        return 0

    anchor_links = [a['href'] for a in found_anchors]

    link_freq_dict = {}
    for link in anchor_links:
        if (not link in link_freq_dict):
            link_freq_dict[link] = 0
        link_freq_dict[link] += 1
    most_frequent_link = max(link_freq_dict, key=link_freq_dict.get)

    return link_freq_dict[most_frequent_link] / len(anchor_links)

def get_null_link_ratio(soup):

    found_anchors = soup.find_all('a', attrs={"href": True})

    if (len(found_anchors) == 0):
        return 0
    
    anchor_links = [a['href'] for a in found_anchors]
    
    null_count = 0
    for link in anchor_links:
        if link == '#' or link == '#content':
            null_count += 1

    return null_count / len(anchor_links);


def get_html_data(url, result):

    html_data = {}

    soup = BeautifulSoup(result.text, "html.parser")
    print("\tsoup")

    print("\thas_iframe")
    found_iframe = soup.find('iframe')
    html_data['has_iframe'] = found_iframe != None

    print("\tmost_frequent_domain")
    current_domain = urlparse(url).netloc
    html_data['most_frequent_domain_in_anchor'] = is_domain_more_frequent_in_tag(soup, current_domain, 'a')
    html_data['most_frequent_domain_in_link'] = is_domain_more_frequent_in_tag(soup, current_domain, 'link')
    html_data['most_frequent_domain_in_source'] = is_domain_more_frequent_in_tag(soup, current_domain, 'source')

    print("\tcommon_page_rate")
    html_data['common_page_rate'] = get_page_detection_rate(soup)

    print("\tfooter_common_page_rate")
    page_footer = soup.find('footer')
    footer_page_rate = 0
    if (page_footer):
        footer_page_rate = get_page_detection_rate(page_footer)
    html_data['footer_common_page_rate'] = footer_page_rate

    print("\thas_anchor_in_body")
    page_body = soup.find('body')
    found_anchor = False
    if page_body:
        found_anchor = page_body.find('a')
    html_data['has_anchor_in_body'] = True if found_anchor else False

    print("\tnull_link_ratio")
    html_data['null_link_ratio'] = get_null_link_ratio(soup)
    footer_null_link_ratio = 0
    if (page_footer):
        footer_null_link_ratio = get_null_link_ratio(page_footer)
    html_data['footer_null_link_ratio'] = footer_null_link_ratio

    return html_data

def ip_to_number(ip_string):
    packedIP = socket.inet_aton(ip_string)
    return struct.unpack("!L", packedIP)[0]

def load_asn_data(file_name):
    asn_data = []
    with open(file_name, "r") as asn_file:
        lines = asn_file.readlines()
        for line in lines:
            tokens = line.split("\t")
            asn_data.append({
                'range_start': ip_to_number(tokens[0]),
                'range_end': ip_to_number(tokens[1]),
                'AS_number': int(tokens[2]),
                'country_code': tokens[3],
            })
    return asn_data

def get_domain_age(domain):
    try:
        whois_response = whois.whois(domain)
        creation_date = whois_response.creation_date

        if (creation_date is None):
            return 0

        if (isinstance(creation_date, list)):
            creation_date = creation_date[0]
        start = creation_date.timestamp()
        end = datetime.datetime.now().timestamp()
        return end - start

    except whois.parser.PywhoisError:
        return 0

def get_domain_page_rank(domain):

    result = requests.get(
        "https://openpagerank.com/api/v1.0/getPageRank",
        headers={
            "API-OPR": "gwo4kk0oow4sok8gokwwsccc08swckcc8884ggs4",
        },
        params={
          "domains[]": [domain]
        }
    )

    if (result.ok):
        result_data = result.json()
        page_rank = result_data['response'][0]['page_rank_decimal']
        if not page_rank:
            return 0
        return page_rank

    return 0