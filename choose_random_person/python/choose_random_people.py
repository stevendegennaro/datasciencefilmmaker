import pandas as pd
import numpy as np
import fiona
from shapely.geometry import Polygon, Point, shape
import matplotlib.pyplot as plt
import sys

# Function to randomly choose a latitude and longitude within a given ploygon
def get_random_location(polygon: Polygon) -> Point:
	# Find the northest, southest, eastest, and westest points in the block
	min_x, min_y, max_x, max_y = polygon.bounds
	# Draw a random point within those bounds
	point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
	# If the point is not within the polygon:
	while not point.within(polygon):
		# Keep drawing until it is
		point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
	return point    

def load_hash(seed:int = None) -> None:
	# If the user provides a seed, use it.
	if seed:
		np.random.seed(seed)
	print("Loading hash")
	# Load the hash table, but only if it's not loaded already
	if 'hash_df' not in globals():
		global hash_df
		hash_df = pd.read_csv('data/hash.csv',dtype={'FIPS': str})

def get_random_people(n_samples:int = 25) -> pd.DataFrame:

	print(f"Drawing {n_samples} samples")
	sample = hash_df.sample(n_samples,weights='population',replace=True)
	sample['polygon'] = None
	sample['location'] = None

	print("Drawing locations")
	for index,person in sample.iterrows():       # For each of the blocks we just drew
		# Open the file for that person's state and get the boundaries of the person's block
		state = person['FIPS'][:2]
		filename = 'data/TIGER/tl_2020_' + state + '_tabblock20/tl_2020_' + state + '_tabblock20.shp'
		with fiona.open(filename, 'r') as source:
			# print(source[int(person['line'])]['geometry']['coordinates'])
			polygon = shape(source[int(person['line'])]['geometry'])
			# print(polygon)

		# Check to make sure we didn't screw anything up
		assert type(polygon) == Polygon, f"Something is wrong.\n{person}, {polygon}"

		# Add this location to our list
		sample.loc[index,'polygon'] = polygon
		sample.loc[index,'location'] = get_random_location(polygon)
	return sample

def plot_location(person: pd.DataFrame, ax: plt.Axes,color="black") -> None:
	x,y = person['polygon'].exterior.xy
	ax.plot(x,y,color=color)
	ax.plot(person['location'].x,person['location'].y,marker="+",color=color)

def test_blocks() -> None:
	load_hash()
	sample_df = get_random_people(50)

	fig,axes = plt.subplots(5,5)
	plt.subplots_adjust(wspace=0, hspace=0)
	for i in range(5):
		for j in range(5):
			person = sample_df.iloc[5*i+j]
			plot_location(person,axes[i][j])
			axes[i][j].set_xticks([])
			axes[i][j].set_yticks([])

	fig.suptitle("Randomly Chosen Census Blocks with\nRandomly Chosen Position Within Block")
	plt.show()

def test_shape(person,n):
	x,y = person['polygon'].exterior.xy
	locations = [get_random_location(person['polygon']) for _ in range(n)]
	xs = [location.x for location in locations]
	ys = [location.y for location in locations]
	plt.plot(x,y,color="black")
	plt.scatter(xs,ys,marker=".",s=1,color="blue")
	plt.title("Random Locations Drawn from Within a Block")
	plt.xlabel("Longitude")
	plt.ylabel("Latitude")
	plt.show()