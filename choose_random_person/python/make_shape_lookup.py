import pandas as pd 
import geopandas as gpd

# Uses the blocks.csv file and the shape files for each state and
# creates a lookup table so that we can look up the shapefile
# quickly by row number if we know the FIPS #.

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