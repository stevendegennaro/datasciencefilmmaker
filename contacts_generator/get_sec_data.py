import requests
import json

API_KEY = 'cafc16655d477253b856fa133948f420fe2b0a8a3f53ac10e0816d92db7e8d66'

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
