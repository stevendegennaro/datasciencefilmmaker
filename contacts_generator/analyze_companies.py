import json
from collections import Counter
import numpy as np

def get_company_names():
    exchanges = ['nyse','nasdaq']
    names = []
    for exchange in exchanges:
        filename = 'data/' + exchange +'.json'
        with open(filename, "r", encoding='utf-8') as f:
            data = json.load(f)
        names.extend([company['name'].title() for company in data])
        names = sorted(list(set(names)))
        ##### Clean the names to remove ones that have weird punctuation ######
    return names

fullnames = get_company_names()

suffixes = []
for name in fullnames:
    suffixes.append(name.split()[-1])

suffix_counts = Counter(suffixes)

# remove common suffixes
# remove common suffixes
common_suffixes = {}
for key in suffix_counts:
    if suffix_counts[key] > 12:
        common_suffixes[key] = suffix_counts[key]

names = [name.split() for name in fullnames]
for i in range(len(names)):
    if names[i][-1] in common_suffixes.keys():
        names[i] = names[i][:-1]
    newname = ''
    for w in names[i]:
        newname = ' '.join([newname,w])
        # print(newname)
    names[i] = newname.strip()

def get_company_name():
    name = np.random.choice(names)
    if len(name.split()) > 4:
        name = np.random.choice(names)
    return name

# for _ in range(100):
#   print(get_company_name())
