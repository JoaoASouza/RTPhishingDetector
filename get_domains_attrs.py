from datahelper import *

domains_data = []

OUT_FILE_NAME = "out" + datetime.today().strftime('%Y-%m-%d') + "_legitimate.csv"

with open("lists/legitimate-domains.txt") as domains_file:
    domains = domains_file.readlines()
    for domain in domains:
        domain = domain[:-1]

        is_https = True

        response = check_domain(domain, "https")
        if (response == 0):
            is_https = False
            response = check_domain(domain, "http")

        if (response == 0 or not response):
            continue

        url = response.url

        domain_data = {}

        domain_data['url'] = url
        domain_data['url_length'] = len(url)
        domain_data['is_ip'] = is_valid_ip(domain)
        domain_data['url_has_at_char'] = '@' in url
        domain_data['number_of_dots'] = url.count('.')
        domain_data['is_https'] = is_https
        domain_data['domain_age'] = get_domain_age(domain)
        domain_data['page_rank'] = get_domain_page_rank(domain)
        domain_data['url_redirection'] = url.count("//") > 1
        domain_data['domain_has_dash_char'] = '-' in domain
        domain_data['url_has_https_token'] = False if 'https' not in url else url.index('https') != 0
        
        dns_data = get_dns_data(domain)
        domain_data.update(dns_data)

        html_data = get_html_data(url)
        domain_data.update(html_data)

        domains_data.append(domain_data)

        print(domains_data)

        print("writing...")
        write_data_to_file(domains_data, "datasets/" + OUT_FILE_NAME)
        print("writing done!")

        
