
import re
import tldextract
import ipaddress

def extract_url_features(url):
    ext = tldextract.extract(url)
    domain = ext.domain + '.' + ext.suffix
    subdomain = ext.subdomain
    try:
        is_ip = ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    specials = sum(not c.isalnum() for c in url)

    features = {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "TLDLength": len(ext.suffix),
        "NoOfSubDomain": subdomain.count('.') + (1 if subdomain else 0),
        "IsDomainIP": is_domain_ip,
        "NoOfLettersInURL": letters,
        "NoOfDegitsInURL": digits,
        "LetterRatioInURL": round(letters / len(url), 3),
        "DegitRatioInURL": round(digits / len(url), 3),
        "NoOfEqualsInURL": url.count('='),
        "NoOfQMarkInURL": url.count('?'),
        "NoOfAmpersandInURL": url.count('&'),
        "SpacialCharRatioInURL": round(specials / len(url), 3),
        "Bank": int("bank" in url.lower()),
        "Pay": int("pay" in url.lower()),
        "Crypto": int("crypto" in url.lower())
    }

    return features


url = "http://www.bank-login.example.com?verify=account&id=123"
features = extract_url_features(url)
print(features)
