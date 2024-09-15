import pandas as pd
import numpy as np
import calendar
import jdcal

################################
##### Import Eclipse Data ######
################################

def convert_time_to_julian_date_fraction(time_string: str) -> float:
    h,m,s = time_string.split(':')
    return (60 * 60 * int(h) + 60 * int(m) + int(s))/(24*60*60)

def calculate_julian_dates(data: pd.DataFrame) -> list[float]:
    julian_date_list = []
    for index, row in data.iterrows():
        if row['Year'] < 1582 or (row['Year'] == 1582 and row['Month'] < 10):
            julian_date = jdcal.jcal2jd(row['Year'],row['Month'],row['Day'])
        else:
            julian_date = jdcal.gcal2jd(row['Year'],row['Month'],row['Day'])
        julian_date = julian_date[0]-0.5 + julian_date[1]
        julian_date += convert_time_to_julian_date_fraction(row['Time'])
        julian_date_list.append(julian_date)
    return julian_date_list

def get_level_and_variance(data):
    time_to_next = np.array(data['Time to Next'])
    time_to_next[time_to_next > 170] = 200
    time_to_next[(time_to_next < 170) & (time_to_next > 140)] = 100
    time_to_next[time_to_next < 99] = 0
    time_to_next /= 100
    time_to_next = time_to_next.astype(np.int32)
    data['Level'] = time_to_next

    means = data.groupby('Level')['Time to Next'].mean()
    data['Variance'] = np.array(data['Time to Next']) - np.array(means[data['Level']])

def import_solar_eclipse_data() -> pd.DataFrame:
    filename = 'data/eclipse.gsfc.nasa.gov_5MCSE_5MCSEcatalog_no_header.txt'

    data = pd.read_fwf(filename,header=None,infer_nrows=10000)
    data.set_index(0,inplace=True)
    data.index.name = 'Cat. Number'
    data.columns = ('Canon Plate','Year','Month','Day','Time',\
                    'DT','Luna Num','Saros Num','Type','Gamma','Ecl. Mag.',\
                    'Lat.','Long.','Sun Alt','Sun Azm','Path Width','Central Duration')

    #### Fix dates ###
    # Convert month abbreviations to numbers
    month_abbr = [m for m in calendar.month_abbr]
    data['Month'] = data['Month'].map(lambda m: month_abbr.index(m)).astype('Int8')
    # Convert to Julian Dates
    data['Julian Date'] = calculate_julian_dates(data)

    # Calculate intervals between eclipses
    data['Time to Next'] = pd.DataFrame(data['Julian Date'].diff()).shift(-1)
    data = data[:-1]

    get_level_and_variance(data)

    return data