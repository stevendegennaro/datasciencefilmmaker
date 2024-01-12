from bs4 import BeautifulSoup
import requests
import string
import json
import time

website = "https://www.baseball-reference.com/players/"
alphabet = list(string.ascii_lowercase)
filename="data/all_names.json"

with open(filename,"w") as f:
    f.write('[\n')
    for letter in alphabet:
        url = website + letter
        print(url)
        soup = BeautifulSoup(requests.get(url).text, 'html5lib')
        names = [p.text.strip() for p in soup.findAll('div',attrs={'id':'div_players_'})]
        names = names[0].split('\n')
        for name in names:
            f.write('"' + name +'",\n')
        time.sleep(3.5)
    f.write(']')