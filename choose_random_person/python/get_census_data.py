import requests
import json
import pandas as pd
import sys
import os
import time
from collections import Counter
from matplotlib import pyplot as plt
import requests
from bs4 import BeautifulSoup
import geopandas as gpd

#### Main function for downloading the census data, 
#### first by state, then by county, then by block
def get_census_data():
    with open('census_api_key.txt') as f:
        API_KEY = f.readline()

    BASE_URL = 'https://api.census.gov/data/2020/dec/pl'

    # Check if we already have a list of states and their populations
    if os.path.isfile('data/states.csv'):
        # If so, read it in
        states_df = pd.read_csv('data/states.csv',dtype={'state': str})
        states_df.set_index('state', inplace=True)
    # If not, get it
    else:
        print("Downloading state-level data")
        PARAMS = {
            'get': 'NAME,P1_001N',
            'for': 'state:*',
            'key': API_KEY
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

        # Output the dataframe to a file
        states_df.to_csv('data/states.csv')

    # Check if we already have a list of counties and their populations
    if os.path.isfile('data/counties.csv'):
        # If so, read it in
        counties_df = pd.read_csv('data/counties.csv',dtype={'state': str,'county':str})
        counties_df.set_index(['state', 'county'], inplace=True)

    # If not, make one
    else:
        print("Downloading county-level data")
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

    # Check to see if the total state populations match the summed populations of the counties
    assert (counties_df.groupby(level='state').sum()['population'] - states_df['population'] == 0).all()

    # Check if the blocks file has already been created, if not, create it
    if not os.path.isfile('data/blocks.csv'):
        with open('data/blocks.csv','w') as f:
            f.write('FIPS,population\n')

    # If it already exists, get the last entry so we can continue from there
    else:
        print("Downloading block-level data")
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
        # This shouldn't be necessary if you are using an API key. If not, the API
        # will limit the number of calls in a certain period of time
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
            # Ideally, this shouldn't happen either
            print(response.status_code)
            sys.exit()

### Tests the census data that we downloaded for internal consistency,
### then draws random blocks from the data weighted by population and compares
### them to the actual populations of those blocks. If working properly,
### this should just be a straight line though the origin (with scatter)
def test_census_data():

    # Read in the dataframes for the states and the blocks
    blocks_df = pd.read_csv('data/blocks.csv',dtype={'FIPS': str})
    states_df = pd.read_csv('data/states.csv',dtype={'state': str})
    states_df.set_index('state', inplace=True)

    # Check to see that the total of blocks for each state matches the state population
    # if not, output error message and quit
    print("Checking totals")
    for STATE_FIPS in states_df.index:
        print(STATE_FIPS)
        assert blocks_df.loc[blocks_df['FIPS'].str.startswith(STATE_FIPS),'population'].sum() \
                == states_df.loc[STATE_FIPS,'population'], \
                f"Populations for {states_df.loc[STATE_FIPS,'NAME']} do not match:  \
                {blocks_df.loc[blocks_df['FIPS'].str.startswith(STATE_FIPS),'population'].sum()} \
                {states_df.loc[STATE_FIPS,'population']}"

    total_pop = blocks_df['population'].sum()

    n_samples = total_pop
    print(f"Generating {total_pop} Random Cameras")

    # Create random sample based on block populations
    counter = dict(Counter(blocks_df.sample(n_samples,weights='population',replace=True)['FIPS']))

    # Convert counter object to a dataframe
    rows = []
    for key, value in counter.items():
        row = {'FIPS': key, 'count': value}
        rows.append(row)
    n_cameras = pd.DataFrame(rows)

    # join the two dataframes so I can compare 
    # 'population' to 'count' for each block
    print("joining")
    n_cameras.set_index('FIPS',inplace=True)
    blocks_df.set_index('FIPS',inplace=True)
    joined = blocks_df.join(n_cameras)
    joined.fillna(0, inplace = True)
    joined['count']=joined['count']

    # plot it
    plt.scatter(joined['population'],joined['count'],marker=".",color="black",s=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

### Function for downloading all of the shapefules from census 2020 ###
def download_shape_files():
    print("Downloading shape files")
    site_address = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
    response = requests.get(site_address)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        links = [a['href'] for a in table.find_all('a') if a.has_attr('href')  and a['href'].startswith('tl_')]

    states_df = pd.read_csv('data/states.csv',dtype={'state': str})
    state_codes = list(states_df['state'])

    for link in links[1:]:
        if link[8:10] in state_codes:               # Skip the ones we don't need
            url = site_address + link
            print(f"Downloading {link}")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for HTTP failures

            # Open local file for writing the content of the ZIP file
            with open("data/TIGER/" + link, "wb") as file:
                for chunk in response.iter_content(chunk_size=819200):
                    file.write(chunk)

# Uses the blocks.csv file and the shape files for each state and
# creates a lookup table so that we can look up the shapefile
# quickly by row number if we know the FIPS #.
def make_shape_lookup():

    print("Generating lookup table")
    blocks_df = pd.read_csv('data/blocks.csv',dtype={'FIPS': str})
    blocks_df.set_index('FIPS', inplace=True)
    states_df = pd.read_csv('data/states.csv',dtype={'state': str})
    states_df.set_index('state', inplace=True)

    for state in states_df.index[2:]:
        print(state)
        
        # Read in the shapefile for this state
        filename = 'data/TIGER/tl_2020_' + state + '_tabblock20/tl_2020_' + state + '_tabblock20.shp'
        gdf = gpd.read_file(filename)

        # Combine the state, county, tract, and block info to make 15-digit FIPS
        gdf['FIPS'] = gdf['STATEFP20']+gdf['COUNTYFP20']+gdf['TRACTCE20'] + gdf['BLOCKCE20']
        
        # The index number is the line # in the shape file
        gdf.reset_index(inplace=True)
        gdf = gdf.rename(columns = {'index':'line'})
        
        # Create a new df with just the info we want
        lookup_df = gdf.loc[:,['FIPS','line']]
        
        # Set index to FIPS and sort
        lookup_df.set_index('FIPS',inplace=True)
        lookup_df.sort_index(inplace=True)
        
        # Join with the blocks_df (which contains the population info)
        lookup_df = lookup_df.join(blocks_df,how='inner')
        print(lookup_df)
        
        # write (or append) to the file
        if state == '01':
            lookup_df.to_csv('data/lookup.csv',mode='w')
        else:
            lookup_df.to_csv('data/lookup.csv',mode='a',header=False)

get_census_data()
test_census_data()
download_shape_files()
make_shape_lookup()

