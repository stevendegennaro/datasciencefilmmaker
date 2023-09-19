from matplotlib import pyplot as plt
import pandas as pd

# blocks_df = pd.read_csv('data/blocks.csv',dtype={'FIPS': str})

def block_population_hist(blocks_df):
	plt.hist(blocks_df['population'], bins=200,color="midnightblue",edgecolor="k")
	plt.yscale('log')
	plt.xlabel('Population')
	plt.ylabel('Log(N)')
	plt.title('Distribution of Block Populations in 2020 Census')
	plt.show()