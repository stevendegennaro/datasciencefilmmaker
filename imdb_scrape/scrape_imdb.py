import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from lxml import html
import requests
import json
import sys
from pprint import pprint
import ast
from denumerize import denumerize
import traceback
import re
import time
import os


# Workflow:
#   scrape_letterboxd_lists() -> letterboxdhorror.csv, letterboxdthriller.csv ->
#   get_lb_info() -> culledimdblinks.csv ->
#   populate_list_details() -> imdbinfoformatted.json


# pd.set_option('display.max_columns', None)
genres = ['horror','thriller']
n_pages = [509,392]             # This was true when I ran the code. Found manually.

### Function to scrape Letterboxd lists by genre and put into a csv
### This function uses Selenium to get teh eweb page data because
### the pages that we are scraping are created synamically at
### run time via javascript, so we have to wait for them to load
### completely before scraping

def scrape_letterboxd_lists():

    # Open the virtual browser
    browser = webdriver.Chrome(ChromeDriverManager().install())

    # For each genre
    for g in range(len(genres)):

        # Open a file for that genre
        filename = 'data/letterboxd' + genres[g] + '.csv'
        with open(filename, 'w') as file_object:

            page = 1 # Start at page 1
            total = 0 # 

            # Get every page
            while page <= n_pages[g]:

                print(f"Starting page number {page}...")
                pagename = f'https://letterboxd.com/films/popular/genre/{genres[g]}/size/small/page/{page}/'
                browser.get(pagename)
                print("Getting innerHTML...")
                innerHTML = browser.execute_script("return document.body.innerHTML")
                print("Creating tree...")
                tree = html.fromstring(innerHTML)

                # Grab the relevant data and put into lists
                print("parsing...")
                newtitles = tree.xpath('//ul[@class="poster-list -p70 -grid"]/li/div/@data-film-name')
                newurls = tree.xpath('//ul[@class="poster-list -p70 -grid"]/li/div/@data-film-link')
                newyears = tree.xpath('//ul[@class="poster-list -p70 -grid"]/li/div/@data-film-release-year')
                newids = tree.xpath('//ul[@class="poster-list -p70 -grid"]/li/div/@data-film-id')

                if ((len(newtitles) != 72) & (not (page>=n_pages[g]))):
                    print("Glitch")     # Page did not load properly so go back to the 
                    continue            # top of the loop without incrementing and try again
                else:
                    page += 1
                    total += len(newtitles)
                    print("writing to file...")
                    print(f"{page} {total}")
                    for i in range(0,len(newtitles)):
                        file_object.write(f'"{newids[i]}",')
                        file_object.write(f'"{newtitles[i]}",')
                        file_object.write(f'"https://letterboxd.com{newurls[i]}",')
                        file_object.write(f'"{newyears[i]}"\n')

        print(f"File {filename} written")

### Go to each individual movie's letterboxd page and get the runtime
### Get rid of anything that isn't a feature length film (>70 minutes)
### Also get rid of anything older than 2010, since I am 
### only interested in recent movies and active companies
### Then get the link to the imdb page for the movie so we can scrape that next
def get_lb_info():

    outfile = 'data/culledimdblinks.csv'

    # For each genre
    for g in range(len(genres)):

        # File names
        infile = f'data/letterboxd{genres[g]}.csv'

        # Read in the letterboxd links for that genre and clean data
        df = pd.read_csv(infile,header=None)
        df.columns = ['lb_id','title','lb_url','year']

        # If it doesn't have a release year, set the year to 0
        # Then set the type for the year column to 32-bit integer
        df['year'] = df['year'].fillna(0).astype('int32')

        # Sort by year and get rid of anything older than 2010 (including 0)
        # Or newer than 2022 (The year I created this list)
        df = df.loc[(df['year'] >= 2010) & (df['year'] < 2023)]
        df.sort_values(by = 'year',ascending = False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Columns for data we are going to add:
        df['genre'] = genres[g].title()
        df['run_time'] = None
        df['imdb_url'] = None
        df['lb_stars'] = None

        if g == 0:
            df.columns.to_frame().T.to_csv(outfile, index = False, mode = "w", header = False)
            ids = []            # So we can remove duplicates when we do the next genre
        # For each movie:
        for index,row in df.iterrows():
            print(index, "\n", row)
            if row['lb_id'] not in ids:
                # Open the letterboxd page for this title
                pagename = row['lb_url']
                page = requests.get(pagename)
                tree = html.fromstring(page.content)

                # Get the run_time and clean it up to get # of minutes
                run_time = tree.xpath('//p[@class="text-link text-footer"]/text()')
                for i,s in enumerate(run_time):
                    run_time[i] = run_time[i].__str__()
                    run_time[i] = ''.join(run_time[i].split())
                try:
                    run_time = run_time[0]
                except:
                    continue
                if(run_time[0]=="M"):
                    continue
                run_time = int(run_time.replace("minsMoreat","").replace("minMoreat",""))

                # Get rid of anything with a runtime less than 70 minutes (feature length)
                if(run_time < 70):
                    continue
                else:
                    df.loc[index,'run_time'] = run_time
                print(f"Run Time = {df.loc[index,'run_time']}")

                # Get the imdb link
                imdb_url=tree.xpath('//p[@class="text-link text-footer"]/a[@data-track-action="IMDb"]/@href')
                if imdb_url:
                    df.loc[index,'imdb_url'] = imdb_url[0].__str__()
                else:
                    continue
                print(F"IMDb link = {df.loc[index,'imdb_url']}")

                lb_stars = tree.xpath('//meta[@name="twitter:data2"]/@content')
                if lb_stars != []:
                    if len(lb_stars[0]) >= 4:
                        df['lb_stars'] = float(lb_stars[0][:4])

                # Output each line
                df.loc[index].to_frame().T.to_csv(outfile, index = False, mode = "a", header = False)

                ids.append(row['lb_id'])
            else:
                print("skipping...")

### Helper function for populate_list_details()
def get_company_meter(link):

    # Check if we've already gotten the company meter for this company
    # Since reloading webpages via requests is one of the slowest parts of the code
    if link not in company_meters.index:
        # If not, load the company's IMDb Pro page and get the company meter
        companypage = requests.get(link, cookies=cookies, headers=headers)
        companytree = html.fromstring(companypage.content)
        company_meter = companytree.xpath('//header//ul[@data-testid="const-meter"]//span/text()')
        # Sometimes the page returns as being good but it's not.
        # Reload the page up to 5 times and if it's still no good, return 0
        # And we can deal with it in later subroutines
        n = 0
        while len(company_meter) != 2:
            time.sleep(5)
            n += 1
            if(n > 5): return 0
            print(f"Reloading page {link}")
            companypage = requests.get(link,cookies=cookies, headers=headers)
            companytree = html.fromstring(companypage.content)
            company_meter = companytree.xpath('//header//ul[@data-testid="const-meter"]//span/text()')
        company_meter = denumerize(company_meter[1].upper())
        # Add it to the list
        company_meters[link] = company_meter
    return company_meters[link]

### Main scraper for IMDB info. Uses requests with manual cookies (from Chrome)
### to get title, moviemeter, star rating, genres, MPA rating,
### directors, writers, producers, actors, and their roles,
### and sales, distribution, and production companies and their company meters.
### Puts it all into a dictionary and outputs
def populate_list_details(start = 0):

    global cookies
    global headers
    global company_meters
    company_meters = pd.Series()

    # Cookies were obtained manually from a Chrome session using Inspector
    with open('imdb_cookies.txt','r') as f:
        cookies = ast.literal_eval(f.readline())
        headers = ast.literal_eval(f.readline())

    df = pd.read_csv('data/culledimdblinks.csv')

    ## Append or write to file, depending on if we are starting at the beginning of the list
    if start:
        mode = 'a'
    else:
        mode = 'w'

    # Open file for writing rows that produce exceptions, 
    # so we can rerun them or figure out what's wrong
    errorfile = 'errors.csv'
    if not os.path.exists(errorfile):
        e = open(errorfile,'w')
        e.close()

    with open('data/imdbinfoformatted.json',mode) as outputfile:
        if mode == 'w':
            outputfile.write("[\n")

        for index,row in df[start:].iterrows():
            this_movie = row.to_dict()
            print(this_movie['title'])

            try:
                # Open imdbpro page for this movie
                this_movie['imdb_id'] = this_movie['imdb_url'][26:-12]
                pagename = f"https://pro.imdb.com/title/{this_movie['imdb_id']}/details"
                page = requests.get(pagename,cookies=cookies, headers=headers)
                tree = html.fromstring(page.content)

                # Get title, moviemeter, star rating, genres, MPA rating
                this_movie['imdb_title'] = tree.xpath('//div[@id="title_heading"]/span/span/text()')[0] #title
                this_movie['movie_meter'] = tree.xpath('//div[@id="ranking_graph_container"]//span[@class="a-size-medium aok-align-center"]/text()')
                if this_movie['movie_meter'] == []:
                    this_movie['movie_meter'] = None
                else:
                    this_movie['movie_meter'] = int(this_movie['movie_meter'][0].replace(",","")) #moviemeter
                this_movie['imdb_stars'] = tree.xpath('//div[@id="rating_breakdown"]//span[@class="a-size-medium"]/text()')
                if this_movie['imdb_stars'] == []:
                    this_movie['imdb_stars'] = None
                else:
                    this_movie['imdb_stars'] = float(this_movie['imdb_stars'][0].strip()) #starrating
                this_movie['imdb_genres'] = tree.xpath('((//span[@id="genres"])[1])/a/text()') #genres
                rating = tree.xpath('//span[@id="certificate"]/text()')
                this_movie['rating'] = rating[0].strip() if rating != [] else None #rating

                # Get release details (country of origin, language)
                detailheaders = tree.xpath('//tr[@class="release_details_item"]//th/text()')
                details = tree.xpath('//tr[@class="release_details_item"]//td/text()')
                for i in range(0,len(detailheaders)):
                    detailheader = detailheaders[i].strip().lower()
                    detail = details[i].strip().split(',')
                    strippeddetail = [d.strip() for d in detail]
                    this_movie[detailheader] = strippeddetail
                if not 'languages' in this_movie:
                    this_movie['languages'] = []
                if not 'country of origin' in this_movie:
                    this_movie['country of origin'] = []

                ### I only care about movies that were produced in the US or use English
                ### Some foreign movies that use small amounts of English slip through
                if ('English' in this_movie['languages']) or ('United States' in this_movie['country of origin']):

                    # Financials are  listed with commas and currency symbols, so we strip those out.
                    budget = tree.xpath('//div[@class="a-section a-spacing-small budget_summary"]//text()')
                    if len(budget) > 0:
                        if "$" in budget[1]:
                            this_movie['budget'] = int(budget[1].strip().replace(",","").replace("$",""))
                        else:
                            this_movie['budget'] = budget[1].strip()
                    else:
                        this_movie['budget'] = None
                    
                    domgross = tree.xpath('//div[@class="a-section a-spacing-small gross_usa_summary"]//text()')
                    if len(domgross) > 0:
                        if "$" in domgross[1]:
                            this_movie['domgross'] = int(domgross[1].strip().replace(",","").replace("$",""))
                        else:
                            this_movie['domgross'] = domgross[1].strip()
                    else:
                        this_movie['domgross'] = None
                    
                    worldgross = tree.xpath('//div[@class="a-section a-spacing-small gross_world_summary"]//text()')
                    if len(worldgross) > 0:
                        if "$" in worldgross[1]:
                            this_movie['worldgross'] = int(worldgross[1].strip().replace(",","").replace("$",""))
                        else:
                            this_movie['worldgross'] = worldgross[1].strip()
                    else:
                        this_movie['worldgross'] = None

                    # Directors, Producers, Writers and their actual credited roles
                    # (e.g. "Associate Producer", "Story By", etc.)
                    # We also get each person's imdb link so we can find them easily later
                    pagename = f"https://pro.imdb.com/title/{this_movie['imdb_id']}/filmmakers"
                    page = requests.get(pagename,cookies=cookies, headers=headers)
                    tree = html.fromstring(page.content)

                    directors = tree.xpath('//div[@id="title_filmmakers_director_sortable_table_wrapper"]//tr[@class="filmmaker"]//span[@class="a-size-base-plus"]//a/text()')
                    directorcredits = tree.xpath('//div[@id="title_filmmakers_director_sortable_table_wrapper"]//tr[@class="filmmaker"]//span[@class="see_more_text_collapsed"]/text()')
                    directorlinks = tree.xpath('//div[@id="title_filmmakers_director_sortable_table_wrapper"]//tr[@class="filmmaker"]//span[@class="a-size-base-plus"]//a/@href')
                    this_movie['directors'] = [{'name': name.strip(), 'credit': re.sub(r'\s+', ' ', credit).strip(), 'link': link.strip()} 
                                                for name, credit, link in zip(directors, directorcredits,directorlinks)]

                    writers = tree.xpath('//div[@id="title_filmmakers_writer_sortable_table_wrapper"]//tr[@class="filmmaker"]//span[@class="a-size-base-plus"]//a/text()')
                    writercredits = tree.xpath('//div[@id="title_filmmakers_writer_sortable_table_wrapper"]//tr[@class="filmmaker"]//span[@class="see_more_text_collapsed"]/text()')
                    writerlinks = tree.xpath('//div[@id="title_filmmakers_writer_sortable_table_wrapper"]//tr[@class="filmmaker"]//span[@class="a-size-base-plus"]//a/@href')
                    this_movie['writers'] = [{'name': name.strip(), 'credit': re.sub(r'\s+', ' ', credit).strip(), 'link': link.strip()} 
                                                for name, credit, link in zip(writers, writercredits,writerlinks)]

                    producers = tree.xpath('//div[@id="title_filmmakers_producer_sortable_table_wrapper"]//tr[@class="filmmaker"]//span[@class="a-size-base-plus"]//a/text()')
                    producercredits = tree.xpath('//div[@id="title_filmmakers_producer_sortable_table_wrapper"]//tr[@class="filmmaker"]//span[@class="see_more_text_collapsed"]/text()')
                    producerlinks = tree.xpath('//div[@id="title_filmmakers_producer_sortable_table_wrapper"]//tr[@class="filmmaker"]//span[@class="a-size-base-plus"]//a/@href')
                    this_movie['producers'] = [{'name': name.strip(), 'credit': re.sub(r'\s+', ' ', credit).strip(), 'link': link.strip()} 
                                                for name, credit, link in zip(producers, producercredits,producerlinks)]

                    # Cast (same info as directors etc but also get Star Meter)
                    pagename = f"https://pro.imdb.com/title/{this_movie['imdb_id']}/cast"
                    page = requests.get(pagename,cookies=cookies, headers=headers)
                    tree = html.fromstring(page.content)
                    nactors = len(tree.xpath('//table[@id="title_cast_sortable_table"]//a[@data-tab="cst"]/text()'))
                    actors = tree.xpath('//table[@id="title_cast_sortable_table"]//a[@data-tab="cst"]/text()')
                    actorlinks = tree.xpath('//table[@id="title_cast_sortable_table"]//tr/td[1]//a[@data-tab="cst"]/@href')
                    actorcredits = tree.xpath('//table[@id="title_cast_sortable_table"]//tr//span[@class="see_more_text_collapsed"]/text()')
                    starmeters = tree.xpath(f'//table[@id="title_cast_sortable_table"]//tr//td[@class="a-text-right"]//text()')
                    this_movie['actors'] = [{'name': name.strip(), 
                                             'credit': re.sub(r'\s+', ' ', credit).strip(), 
                                             'link': link.strip(),
                                             'starmeter': int(starmeter.strip().replace(",",""))} 
                                                for name, credit, link, starmeter in zip(actors, actorcredits, actorlinks, starmeters)]

                    # Companies
                    pagename = f"https://pro.imdb.com/title/{this_movie['imdb_id']}/companycredits"
                    page = requests.get(pagename,cookies=cookies, headers=headers)
                    tree = html.fromstring(page.content)

                    print("prodcos")
                    # #production companies
                    companies = tree.xpath('//table[@id="production"]//div[@class="a-section a-spacing-mini a-spacing-top-mini"]//a/text()')
                    links = tree.xpath('//table[@id="production"]//div[@class="a-section a-spacing-mini a-spacing-top-mini"]//a/@href')
                    meters = []
                    # Theses are done in a loop because we have to load each company's page individually
                    for link in links:
                        meters.append(get_company_meter(link))
                    assert len(companies) == len(links) == len(meters), f"prodco numbers do not match \
                                                        {len(companies)} {len(links)} {len(meters)}"
                    this_movie['prodcos'] = [{'name': name.strip(), 
                                              'link': link.strip(),
                                              'company_meter': int(meter)} 
                                                for name, link, meter in zip(companies, links, meters)]

                    # continue
                    print("distributors")
                    # Distributors
                    # Scrape the info
                    companies = tree.xpath('//table[@id="distribution"]//div[@class="a-section a-spacing-mini a-spacing-top-mini"]//a/text()')
                    links = tree.xpath('//table[@id="distribution"]//div[@class="a-section a-spacing-mini a-spacing-top-mini"]//a/@href')
                    formats = tree.xpath('//table[@id="distribution"]//tr//td[3]/div/text()')   
                    regions = tree.xpath('//table[@id="distribution"]//tr//td[2]/div/text()')

                    meters = []
                    for link in links:
                        meters.append(get_company_meter(link))
                    assert len(companies) == len(links) == len(meters) == len(formats) == len(regions), \
                                                    f"disributor numbers do not match \
                                                        {len(companies)} {len(links)} {len(meters)} {len(formats)} {len(regions)}"

                    # Format for output
                    formats = [f.strip() for f in formats.split('|')]
                    this_movie['distributors'] = [{'name': name.strip(), 
                                                   'link': link.strip(),
                                                   'format': Format.strip(),
                                                   'region': region.strip(),
                                                   'company_meter': int(meter)} 
                                                for name, link, Format, region, meter
                                                in zip(companies, links, formats, regions, meters)]

                    print("sales")
                    #### Sales ####
                    companies = tree.xpath('//table[@id="sales"]//div[@class="a-section a-spacing-mini a-spacing-top-mini"]//a/text()')
                    links = tree.xpath('//table[@id="sales"]//div[@class="a-section a-spacing-mini a-spacing-top-mini"]//a/@href')
                    formats = tree.xpath('//table[@id="sales"]//tr//td[3]/div/text()')  
                    regions = tree.xpath('//table[@id="sales"]//tr//td[2]/div/text()')

                    meters = []
                    for link in links:
                        meters.append(get_company_meter(link))
                    assert len(companies) == len(links) == len(meters) == len(formats) == len(regions), \
                                                    f"sales numbers do not match \
                                                        {len(companies)} {len(links)} {len(meters)} {len(formats)} {len(regions)}"
                    # Format for output
                    formats = [f.strip() for f in formats.split('|')]
                    this_movie['sales'] = [{'name': name.strip(), 
                                            'link': link.strip(),
                                            'format': Format.strip(),
                                            'region': region.strip(),
                                            'company_meter': int(meter)} 
                                                for name, link, Format, region, meter
                                                in zip(companies, links, formats, regions, meters)]

                    outputfile.write(json.dumps(this_movie,indent = 4))
                    outputfile.write(",\n")
                    outputfile.flush()

                else:
                    print(f"Skipped... {this_movie['languages']} {this_movie['country of origin']}")

            # Bad HTML or unexpected quirks of individual movies can break the code, so
            # instead of stopping, we just let the user know which titles broke it so that
            # they can be individually inspected
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                print(repr(e), row[1])
                print(traceback.format_exc())
                row.to_frame().transpose().to_csv(errorfile,mode='a',index=False,header=False)  # print to error file
                continue

        outputfile.write("]")

### Main loop ###
# scrape_letterboxd_lists()
# get_lb_info()
# populate_list_details()
