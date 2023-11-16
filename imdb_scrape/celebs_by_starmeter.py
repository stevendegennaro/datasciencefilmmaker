import csv
from lxml import html
from lxml import etree
import requests
import json
import traceback
from io import StringIO, BytesIO

genders = ['male','female']

for gender in genders:
    count = 0
    outputfilename=f'data/starmeter_{gender}.json'
    pagename = f"https://www.imdb.com/search/name/?gender={gender}&start=1"
    with open(outputfilename,'a') as outputfile:
        while count<1000000:
            print(pagename)
            page = requests.get(pagename)
            tree = html.fromstring(page.content)
            entries = tree.xpath('//div[@class="lister-list"]//div[@class="lister-item-content"]')
            for i,entry in enumerate(entries):
                entry = etree.tostring(entry)
                entry = html.fromstring(entry[:1000])
                name = entry.xpath('//h3//a/text()')[0].strip()
                link = entry.xpath('//h3//a/@href')[0][6:]
                job = entry.xpath('//div/p[@class="text-muted text-small"]/text()')
                if job != []:
                    job = job[0].strip()
                else:
                    job = None
                actor = {'name':name,'link':link,'star_meter':count+1,'main_job':job}
                count = count + 1
                print(actor)
                #outputfile.write(f"{actor},\n")        
                outputfile.write(json.dumps(actor))
                outputfile.write(",\n")
            pagename = 'https://www.imdb.com/' + tree.xpath('//div[@class="desc"]//a[@class="lister-page-next next-page"]/@href')[0]