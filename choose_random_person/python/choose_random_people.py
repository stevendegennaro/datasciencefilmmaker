import pandas as pd
import numpy as np
import fiona
import geopandas as gpd
from shapely.geometry import Polygon, Point, shape
import matplotlib.pyplot as plt
import json
import sys
import os
import contextily as cx
import textalloc as ta
import datetime as dt

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
    for index, person in sample.iterrows():
        # For each of the blocks we just drew
        # Open the file for that person's state and 
        # get the boundaries of the person's block
        state = person['FIPS'][:2]
        filename = 'data/TIGER/tl_2020_' + state + '_tabblock20/tl_2020_' + state + '_tabblock20.shp'
        with fiona.open(filename, 'r') as source:
            polygon = shape(source[int(person['line'])]['geometry'])

        # Check to make sure we didn't screw anything up
        assert type(polygon) == Polygon, f"Something is wrong.\n{person}, {polygon}"

        # Add this location to our list
        sample.loc[index,'polygon'] = polygon
        sample.loc[index,'location'] = get_random_location(polygon)

    return sample

# Plots the location of a person
def plot_location(person: pd.DataFrame, ax: plt.Axes,color="black") -> None:
    x,y = person['polygon'].exterior.xy
    ax.plot(x,y,color=color)
    ax.plot(person['location'].x,person['location'].y,marker="+",color=color)

def get_radius():
    return np.random.normal(1200,100)

# main function to create the final file that I need, which is a list of 
# 100 randomly-drawn locations in the U.S. weighted by population
# output in json format so they can be plotted by the Google Maps API
def create_cameras_json(filename: str, n_samples: int = 100):

    if os.path.isfile(filename):
        print(f'File "{filename}" already exists. Exiting.')
        return

    load_lookup()
    cameras = get_random_people(n_samples)
    cameras_dict_list = []
    for index,row in cameras.iterrows():
        try:
            cameras_dict_list.append({'lat':row['location'].y,
                                      'lng':row['location'].x,
                                      'radius':get_radius()})
        except:
            print(row)
            sys.exit()

    with open(filename,'w') as f:
        f.write("[\n")
        for row in cameras_dict_list:
            f.write("\t")
            json.dump(row, f)
            f.write(",\n")
        f.write("]")

def get_philly_metro_polygon():
    metro_file = 'data/TIGER/philly_metro/philly_metro_polygon.shp'
    if os.path.isfile(metro_file):
        philly_df = gpd.read_file(metro_file)
    else:
        filename = 'data/TIGER/tl_2023_us_uac20/tl_2023_us_uac20.shp'
        urban_df = gpd.read_file(filename)
        philly_df = urban_df.loc[[933]]
        philly_df['geometry'] = list(philly_df['geometry'].iloc[0].geoms)[3]
        philly_df.to_file(metro_file)
    return philly_df


def plot_philly_cameras(seed: int = None):

    if not seed:
        seed = np.random.randint(1000000)
        print(f"seed = {seed}")
    np.random.seed(seed)

    BBox = (-76, -74.5, 39.5, 40.5)
    fig, ax = plt.subplots(figsize = (7,7))
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])

    philly_img = plt.imread('data/philly_map.png')
    ax.imshow(philly_img, extent = BBox, aspect = 'auto')
    
    philly_df = get_philly_metro_polygon()
    x,y = philly_df.loc[0,'geometry'].exterior.xy
    plt.plot(x,y,color='black')

    cameras = []
    for i in range(10):
        cameras.append(get_random_location(philly_df.loc[0,'geometry']))
    cameras = gpd.GeoDataFrame(cameras, columns = ['geometry'],crs=philly_df.crs)
    cameras['markersize'] = np.random.randint(50, 800, cameras.shape[0])
    cameras.plot(ax = ax, markersize = cameras['markersize'], color='red',marker = 'o')
    
    murders = []
    for i in range(70):
        murders.append(get_random_location(philly_df.loc[0,'geometry']))
    murders = gpd.GeoDataFrame(murders, columns = ['geometry'],crs=philly_df.crs)
    murders['markersize'] = np.random.randint(10, 300, murders.shape[0])
    murders.plot(ax = ax, markersize = murders['markersize'], color='blue',marker = '.')
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

def random_dates(n = 1, log = True):

    # Start date is day after 2016 election
    start = dt.datetime.strptime('11/9/2016', '%m/%d/%Y')
    end = dt.datetime.today()
    end = dt.datetime.strptime('11/20/2024', '%m/%d/%Y')
    delta = end - start
    # weight so that pace is accelerating
    if log:
        n_days = pd.Series(np.random.default_rng().exponential(scale=delta.days/2, size=n*2))
        n_days = n_days[n_days < delta.days]
        while len(n_days) < n:
            n_days = pd.Series(np.random.default_rng().exponential(scale=delta.days/2, size=n*2))
            n_days = n_days[n_days < delta.days]
        n_days = delta.days - n_days[:n]
    else:
        n_days = pd.Series(np.random.randint(0,delta.days,n))
    n_days = pd.to_timedelta(n_days,'day')
    dates = (n_days + start).dt.strftime('%m/%d/%Y')

    # plt.hist(dates,100)
    # plt.show()
    return dates

def plot_US_cameras(n: int, seed: int = None, hit_fraction = 0.8, add_fraction = 0.2):

    if not seed:
        seed = np.random.randint(1000000)
        print(f"seed = {seed}")
    np.random.seed(seed)

    print("Loading state shapefiles...")
    # Import shapefiles for U.S.
    filename = 'data/TIGER/tl_2023_us_state/tl_2023_us_state.shp'
    state_df = gpd.read_file(filename)
    # Get rid of Hawaii, Alaska, and territories
    state_df = state_df[((state_df['STATEFP'] != '02') & 
                         (state_df['STATEFP'] != '15') & 
                         (state_df['STATEFP'].astype(int) <= 56)
                        )]
    state_df.sort_values("STATEFP",inplace = True)
    # Convert to U.S. National Atas Equal Area
    state_df.to_crs('epsg:2163', inplace = True)

    # Plot state boundaries
    fig, ax = plt.subplots(figsize=(12, 7))
    state_df.boundary.plot(ax = ax,alpha = 0.0)
    # Add map image underneath
    cx.add_basemap(ax, zoom = 5,crs = state_df.crs)

    # Load the lookup table to draw random locations
    load_lookup()
    cameras = get_random_people(n)
    # convert to GeoDataFrame (EPSG:4326 is lat/long coordinates)
    cameras = gpd.GeoDataFrame(cameras, 
                               geometry = cameras.location,
                               crs="EPSG:4326")
    # Convert to U.S. National Atas Equal Area
    cameras.to_crs('epsg:2163', inplace = True)
    # Random marker size
    cameras['markersize'] = np.random.randint(10, 50, cameras.shape[0])

    # Keep x% as locations where there is murder near a camera
    hits = cameras[:int(hit_fraction*len(cameras))]
    hits.reset_index(drop = True, inplace = True)

    # Add some additional murders that aren't near cameras for realism
    add_murders = get_random_people(int(add_fraction*n))
    add_murders = gpd.GeoDataFrame(add_murders, 
                           geometry = add_murders.location,
                           crs="EPSG:4326")
    add_murders.to_crs('epsg:2163', inplace = True)
    murders = pd.concat([hits,add_murders])

    cameras.plot(ax=ax,color='red',marker='o',markersize=cameras['markersize'])
    murders.plot(ax=ax,color='blue',marker='.',markersize=20)
    hits.plot(ax=ax,facecolors='none', edgecolors='black',marker='o',markersize=200)

    x, y = hits['geometry'].x, hits['geometry'].y

    text_list = random_dates(len(x))
    print(len(x),len(text_list))
    ta.allocate_text(fig,ax,x,y,
                    text_list,
                    x_scatter=x, y_scatter=y,
                    textsize=12)

    # for index,row in hits.iterrows():
    #     ax.annotate(text_list.iloc[index],(row['geometry'].x, row['geometry'].y))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

    return(seed)


###########################
#### Testing Functions ####
###########################

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

# Plots a histogram of the populations of every block in the census
def block_population_hist(blocks_df):
    # blocks_df = pd.read_csv('data/blocks.csv',dtype={'FIPS': str})
    plt.hist(blocks_df['population'], bins=200,color="midnightblue",edgecolor="k")
    plt.yscale('log')
    plt.xlabel('Population')
    plt.ylabel('Log(N)')
    plt.title('Distribution of Block Populations in 2020 Census')
    plt.show()