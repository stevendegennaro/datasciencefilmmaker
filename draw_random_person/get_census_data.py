import requests
import json
import pandas as pd
import sys
import os
import time

with open('census_api_key.txt') as f:
	API_KEY = f.readline()

BASE_URL = 'https://api.census.gov/data/2020/dec/pl'

# Check if we already have a list of states and their populations
if os.path.isfile('data/states.csv'):
		states_df = pd.read_csv('data/states.csv',dtype={'state': str})
		states_df.set_index('state', inplace=True)
# If not, get it
else:
	PARAMS = {
		'get': 'NAME,P1_001N',
		'for': 'state:*'
	}
	response = requests.get(BASE_URL, params=PARAMS)
	if response.status_code == 200:
		data = response.json()
		# Make first row the column names
		states_df = pd.DataFrame(data[1:],columns = data[0])
		# Make 'state' row the index (The two-digit code for each state)
		states_df.set_index('state', inplace=True)
		# Change the name of the column
		states_df.rename(columns={'P1_001N':'population'},inplace = True)
		# They return as strings, so make them ints
		states_df['population'] = states_df['population'].astype(int)
		# Sort by state code
		states_df.sort_index(inplace=True)
	else:
		print(f'Error {response.status_code}: {response.text}')

	states_df.to_csv('data/states.csv')


# Check if we already have a list of counties and their populations
if os.path.isfile('data/counties.csv'):
	counties_df = pd.read_csv('data/counties.csv',dtype={'state': str,'county':str})
	counties_df.set_index(['state', 'county'], inplace=True)

# If not, make one
else:
	# Get list of counties in each state:
	counties_df = pd.DataFrame()
	for STATE_FIPS in states_df.index:
		print(states_df.loc[STATE_FIPS])
		PARAMS = {
				'get': 'NAME,P1_001N',
				'for': 'county:*',
				'in': f'state:{STATE_FIPS}',
				'key': API_KEY
		}

		response = requests.get(BASE_URL, params=PARAMS)
		if response.status_code == 200:
			data = response.json()
			counties_df = pd.concat([counties_df,pd.DataFrame(data[1:],columns = data[0])])
		else:
			print(f'Error {response.status_code}: {response.text}')

	# create multiindex from state and county codes
	counties_df.set_index(['state', 'county'], inplace=True)
	counties_df.rename(columns={'P1_001N':'population'},inplace = True)
	counties_df['population'] = counties_df['population'].astype(int)
	# Output to file
	counties_df.to_csv('data/counties.csv')

# Check to see if the state populations match the summed populations of the counties
assert (counties_df.groupby(level='state').sum()['population'] - states_df['population'] == 0).all()

# Check if the blocks file has already been created, if not, create it
if not os.path.isfile('data/blocks.csv'):
	with open('data/blocks.csv','w') as f:
		f.write('FIPS,population\n')

# If it already exists, get the last entry so we can continue from there
else:
	with open('data/blocks.csv','r') as f:
		for line in f:
			pass
		START = counties_df.index.get_loc((line[:2],line[2:5])) + 1

for index, row in counties_df.iloc[START:].iterrows():
	STATE_FIPS = index[0]
	COUNTY_FIPS = index[1]
	print(row)

	PARAMS = {
			'get':'P1_001N',
			'for':'block:*',
			'in':f'state:{STATE_FIPS}%20county:{COUNTY_FIPS}',
			'key': API_KEY
			}

	response = requests.get(BASE_URL, params=PARAMS)
	while response.status_code == 429:
		print("sleeping")
		time.sleep(2)
		response = requests.get(BASE_URL, params=PARAMS)

	if response.status_code == 200:
		data = response.json()
		# Put into dataframe with first row as column names
		blocks_df = pd.DataFrame(data[1:],columns = data[0])
		# combine the individual FIPS codes into one 15-digit code and drop the other columns
		blocks_df['FIPS'] = blocks_df['state'] + blocks_df['county'] + blocks_df['tract'] + blocks_df['block']
		blocks_df.drop(['state','county','tract','block'],axis=1,inplace=True)
		# set the FIPS code as the index
		blocks_df.set_index('FIPS', inplace=True)
		# change pop numbers from string to int
		blocks_df['P1_001N'] = blocks_df['P1_001N'].astype(int)
		# get rid of any rows that have zero population
		blocks_df = blocks_df[blocks_df['P1_001N']!=0]
		blocks_df.to_csv('data/blocks.csv',mode='a',header=False)
		# Check to see if the sum of the blocks is equal to the value for the whole county
		assert row['population'] == blocks_df['P1_001N'].sum(), \
						f"Sums do not match: {row['population']} {blocks_df['P1_001N'].sum()}"
	else:
		print(response.status_code)
		sys.exit()


