import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import sys

# Read in the dataframes for the states and the blocks
blocks_df = pd.read_csv('data/blocks.csv',dtype={'FIPS': str})
states_df = pd.read_csv('data/states.csv',dtype={'state': str})
states_df.set_index('state', inplace=True)

# Check to see that the total of blocks for each state matches the state population
print("Checking totals")
for STATE_FIPS in states_df.index:
	print(STATE_FIPS)
	assert blocks_df.loc[blocks_df['FIPS'].str.startswith(STATE_FIPS),'population'].sum() \
			== states_df.loc[STATE_FIPS,'population'], \
			# if not, output error message and quit
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
