import googlemaps
import json
import time
from typing import Dict

# so as not to expose my api key publicly:
with open('google_api_key.txt') as f:
	API_KEY = f.readline()
gmaps = googlemaps.Client(key=API_KEY)

# Get a list of locations within a given radius of a center
# The actual store names that I used are redacted for the purposes of sharing the code
store = ('<redacted>','<redacted>','<redacted>')
def fetch_places_nearby(location: Dict, radius: float) -> Dict:
	places = []
	# for each of my real stores
	for store in stores:
		print(store)
		page_token = None
		# If there are more pages, keep getting results
		while True:
			result = gmaps.places(query=store, location=location, radius=radius, page_token=page_token)
			places.extend(result.get('results', []))
			page_token = result.get('next_page_token')
			# Wait 2 seconds before attempting to get thenext page
			if page_token:
				time.sleep(2)
			else:
				break
		print(len(places))
	return places

GPS_coords = [(40.092285,         -75.184427),
			  (27.36030470144715, -80.80029185104124),
			  (29.791887448678214, -81.54451746964793),
			  (37.164106958569164, -76.84955430780535),
			  (39.154570866616915, -76.29230258069188),
			  (39.02967942578609, -77.22733438768722),
			  (34.058896151565875, -80.75293145142928),
			  (35.129686818169276, -79.13576782610521),
			  (36.01999561977586, -77.51703731499653),
			  (37.112270929418784, -77.31801308019548),
			  (38.402486734964874, -78.6979144141859),
			  (38.490814912279525, -75.28133171676788),
			  (41.84144606290797, -73.82825572191678),
			  (43.42732614493277, -70.70214637105241),
			  (44.934035208689025, -70.42199231200235),
			  (33.780800015621125, -83.65368949592661),
			  (31.758464809574484, -84.0844942994295),
			  (30.930633185701844, -82.80404707147423),
			  (41.267860490349946, -76.6201279307945),
			  (41.55780122690412, -76.79962440220514),
			  (41.16399671766083, -80.00294604584128),
			  (38.92343258614069, -80.78996903587259),
			  (38.83820538048826, -75.60335796949384),
			  (32.989137164367385, -79.88153228179968),
			  (34.16773823214622, -78.0144055802283),
			  (33.58931554531428, -79.0969051659992),
			  (40.69105549096906, -77.2352797465548),
			  (40.08233069934841, -75.20372544123487)]
radius = 100000

# Figures out if we have included this location already
lats = []
def check_location(lat):
	global lats
	if lat in lats:
		print(f"Already included. {lat}")
		return False
	else:
		lats.append(lat)
		return True

# Main loop to iterate through the different sets of GPS coordinates,
# search for each of the stores, and print out the results in a json file
# that can be read by javascript
filename="stop_n_grab_list.json"
filename="temp.json"
with open(filename,"w") as f:
	f.write("[")
	# For each set of GPS coordinates
	for c,center in enumerate(GPS_coords):
		location = f"{center[0]},{center[1]}"
		print(location)
		# Find all of the stores located nearby
		results = fetch_places_nearby(location, radius)

		# For each of the results
		for i,result in enumerate(results):
			# Check that it has a state listed
			if 'plus_code' in result.keys():
				# Check that we haven't already included this location
				if check_location(result['geometry']['location']['lat']):
					# output the results
					location = {"name":"Stop N Grab",
								 "lat":result['geometry']['location']['lat'],
								 "lng":result['geometry']['location']['lng'],
								 "state":result['plus_code']['compound_code'].split()[2]}
					f.write(json.dumps(location))
					if c != len(GPS_coords) - 1 or i != len(results) - 1: f.write(",")
	f.write("]")


