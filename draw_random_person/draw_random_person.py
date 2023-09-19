import pandas as pd
import numpy as np
import fiona
import geopandas as gpd
from shapely.geometry import Polygon, Point, shape

# Function to randomly choose a latitude and longitude within a given ploygon
def draw_location(polygon):
	# Find the northest, southest, eastest, and westest points in the block
	min_x, min_y, max_x, max_y = polygon.bounds
	# Draw a random point within those bounds
	point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
	# If the point is not within the polygon:
	while not point.within(polygon):
		# Keep drawing until it is
		point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
	return point	

np.random.seed(0)
pd.set_option('display.max_columns', None)


print("Loading hash")
hash_df = pd.read_csv('data/hash.csv',dtype={'FIPS': str})

n_samples = 500
print(f"Drawing {n_samples} samples")
drawn = hash_df.sample(n_samples,weights='population',replace=True)

print("Drawing locations")
locations = []

for i,person in drawn.iterrows():		# For each of the blocks we just drew
	# Open the file for that person's state and get the boundaries of the person's block
	state = person['FIPS'][:2]
	filename = 'data/TIGER/tl_2020_' + state + '_tabblock20/tl_2020_' + state + '_tabblock20.shp'
	with fiona.open(filename, 'r') as source:
		polygon = shape(source[int(person['line'])]['geometry'])

	# Check to make sure we didn't screw anything up
	assert type(polygon) == Polygon, f"Something is wrong.\n{person}, {polygon}"

	# Add this location to our list
	locations.append(draw_location(polygon))

print(locations)
