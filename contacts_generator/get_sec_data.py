import requests
import json

with open('sec_api_key.txt') as f:
    API_KEY = f.readline()

def get_companies_by_exchange(exchange):
    url = f'https://api.sec-api.io/mapping/exchange/{exchange}'
    headers = {
        'Authorization': API_KEY
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Example usage
nasdaq_companies = get_companies_by_exchange(nasdaq)
nyse_companies = get_companies_by_exchange('nyse')

filename = 'nyse.json'
with open(filename, "w") as f:
    json.dump(nyse_companies,f)


filename = 'nasdaq.json'
with open(filename, "w") as f:
    json.dump(nasdaq_companies,f)
