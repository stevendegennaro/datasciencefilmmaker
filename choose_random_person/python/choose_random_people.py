import pandas as pd
import numpy as np
import fiona
from shapely.geometry import Polygon, Point, shape
import matplotlib.pyplot as plt
import json
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

# Function to load the lookup table created by make_shape_lookup.py
# Also sets the random seed (if provided)
def load_lookup(seed:int = None) -> None:
    # If the user provides a seed, use it.
    if seed:
        np.random.seed(seed)
    print("Loading lookup table...")
    # Load the lookup table, but only if it's not loaded already
    if 'lookup_df' not in globals():
        global lookup_df
        lookup_df = pd.read_csv('data/lookup.csv',dtype={'FIPS': str})

# Function to use the lookup table to draw a random group of 
# lattitudes and longitudes in the U.S., weighted by population 
# Requires call to load_lookup() first
def get_random_people(n_samples:int = 25) -> pd.DataFrame:

    global lookup_df

    print(f"Drawing {n_samples} samples")
    sample = lookup_df.sample(n_samples,weights='population',replace=True)
    sample['polygon'] = None
    sample['location'] = None

    print("Drawing locations")
    for index,person in sample.iterrows():
        # For each of the blocks we just drew
        # Open the file for that person's state and 
        # get the boundaries of the person's block
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

# Plots the location of a random person
def plot_location(person: pd.DataFrame, ax: plt.Axes,color="black") -> None:
    x,y = person['polygon'].exterior.xy
    ax.plot(x,y,color=color)
    ax.plot(person['location'].x,person['location'].y,marker="+",color=color)

# Test function to make sure we are drawing random blocks
def test_blocks() -> None:
    load_lookup()

    # Get 25 random census blocks
    sample_df = get_random_people(25)

    # Plot the shapes of the blocks in a 5 x 5 grid
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

# Tests whether we are drawing randomly from within a block properly
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

# main function to create the final file that I need, which is a list of 
# 100 randomly-drawn locations in the U.S. weighted by population
# output in json format so they can be plotted by the Google Maps API
def create_cameras_json(n_samples: int = 100):
    load_lookup()
    cameras = get_random_people(n_samples)
    cameras_dict_list = []
    for index,row in cameras.iterrows():
        try:
            cameras_dict_list.append({'lat':row['location'].y,
                                      'lng':row['location'].x,
                                      'radius':np.random.normal(1200,100)})
        except:
            print(row)
            sys.exit()
    with open('../html and javascript/camera_list.json','w') as f:
        json.dump(cameras_dict_list,f)


