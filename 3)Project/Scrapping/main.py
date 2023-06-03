import requests
import os
from bs4 import BeautifulSoup
from numba import cuda

url1 = "https://www.1mg.com"
url2 = "https://www.apollopharmacy.in/"
url3 = "https://healthplus.flipkart.com/"
url4 = "https://pharmeasy.in/"
testurl = "https://webscraper.io/test-sites/e-commerce/allinone"

@cuda.jit
def fetchAndSaveToFile(url, path):
    try:
        http_proxy = "83.79.50.233"
        https_proxy = "83.79.50.233"
        ftp_proxy = "83.79.50.233"
        
        proxies = {"http": http_proxy, "https": https_proxy, "ftp": ftp_proxy}
        
        r = requests.get(url, proxies=proxies)
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"Successfully fetched and saved data from {url} to {path}")
        print(r.status_code)
    except requests.exceptions.RequestException as e:
        print(f"Failed to make a request to {url} with proxies")
        print(e)
    except Exception as e:
        print("An error occurred:")
        print(e)


# Launch CUDA kernel on the GPU
fetchAndSaveToFile[1, 1](url2, "data/apollopharma.html")
