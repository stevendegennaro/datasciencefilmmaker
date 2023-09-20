import requests
from bs4 import BeautifulSoup
import pandas as pd

### Helper app for downloading all of the shapefules from census 2020 ###

site_address = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
response = requests.get(site_address)
if response.status_code == 200:
	soup = BeautifulSoup(response.text, 'html.parser')
	table = soup.find('table')
	links = [a['href'] for a in table.find_all('a') if a.has_attr('href')  and a['href'].startswith('tl_')]

states_df = pd.read_csv('data/states.csv',dtype={'state': str})
state_codes = list(states_df['state'])

for link in links[1:]:
	if link[8:10] in state_codes:				# Skip the ones we don't need
		url = site_address + link
		print(f"Downloading {link}")
		response = requests.get(url, stream=True)
		response.raise_for_status()  # Raise an error for HTTP failures

		# Open local file for writing the content of the ZIP file
		with open("data/TIGER/" + link, "wb") as file:
			for chunk in response.iter_content(chunk_size=819200):
				file.write(chunk)