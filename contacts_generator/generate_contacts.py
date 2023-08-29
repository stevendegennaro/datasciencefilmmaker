### Code for generating fake contacts to be used in a movie ###
import pandas as pd
import numpy as np
import analyze_contacts
import sys
from pprint import pprint
from analyze_companies import get_company_name

# np.random.seed(1)

frequencies = analyze_contacts.get_frequencies()
emails = analyze_contacts.get_emails()
area_codes = analyze_contacts.get_area_codes()
domains = [email.split('@')[1] for email in emails]

id_prob =  pd.Series({  "first":0.9,
						"dot_first":0.2,
						"middle":0.4,
						"dot_last":0.6,
						"last":0.5,
						"initial_first":0.9,
						"initial_middle":0.9,
						"initial_last":0.1,
						"swap":0.2
					  })

def generate_email(entry):
	domain = np.random.choice(domains)
	while 'film' in domain \
	   or 'production' in domain \
	   or 'notification' in domain \
	   or 'fest' in domain \
	   or 'sagaftra' in domain:
		domain = np.random.choice(domains)
	first = full_names["First"][entry]
	middle = full_names["Middle"][entry]
	last = full_names["Last"][entry]

	include = pd.Series([True for n in range(5)],index = id_prob.index[:5])

	def get_identifier(elements):
		identifier = ''
		for n in range(5):
			if include[n]:
				identifier += elements[n]
		return identifier

	# Randomize the other elements
	id_random =  pd.Series(np.random.random(9), index = id_prob.index)

	# Swap the first and last names
	if id_random["swap"] < id_prob["swap"]:
		identifier = get_identifier([last,'.','','',first])
		return identifier + '@' + domain

	include = [True  if (id_random[n] < id_prob[n]) else False for n in range(5)]
	include = pd.Series(include,index = id_prob.index[:5])

	# If the first name is not inlcuded
	if not include["first"]:
		include["middle"] = False
		include["last"] = True
		include["dot_first"] = False
		include["dot_last"] = False

	# It the last name is not included
	if not include["last"]:
		include["dot_last"] = False

	# If the middle name is not included
	if not include["middle"]:
		if include["dot_first"]:
			include["dot_last"] = False
		if not include["last"]:
			include["dot_first"] = False
			include["dot_last"] = False

	# Use first initial instead of name
	if (id_random["initial_first"] < id_prob["initial_first"]):
		first = first[0]
		middle = middle[0]
		include["last"] = True

	# Use middle initial instead of name
	if (id_random["initial_middle"] < id_prob["initial_middle"]):
		middle = middle[0]
		if include["middle"] and include["last"]:
			if include["dot_first"]: include["dot_last"] = True

	elements = [first,'.',middle,'.',last]
	identifier = get_identifier(elements)
	if domain in ['gmail.com','yahoo.com','hotmail.com','icloud.com','ymail.com']:
		digits = np.random.choice([9,99,9999])
		identifier = identifier + str(np.random.randint(digits))
	return identifier + '@' + domain

def generate_company():
	return get_company_name()

def generate_company_email(email,company):
	identifier = email.split('@')[0]
	domain = company.split()[0].lower() + '.com'
	return identifier + '@' + domain 

def generate_phone_number():
	return (('+1 ' if np.random.random() < 0.4 else '') +  '(' +
							  np.random.choice(area_codes) + ') ' +
							  str(np.random.randint(100, 999)) + '-' +
							  str(np.random.randint(0,9999)).zfill(4))


#import last names (taken from Census 2010 data)
ln_file = "data/Names_2010Census.csv"
lastnames_df = pd.read_csv(ln_file)
#drop the last row ("All other names")
lastnames_df.drop(index=lastnames_df.tail(1).index,axis=0,inplace=True)
lastnames_df.loc[lastnames_df['name'].isna(),'name'] = "Null"
lastnames_df['name'] = lastnames_df['name'].str.title()

# import first names (taken from SSA most common names for 1985)
fn_file = "data/firstnames.csv"
firstnames_df = pd.read_csv(fn_file,header=[0],index_col=[0])
#separate them into male and female first names
firstnames_male = firstnames_df.iloc[:,0:2]
firstnames_female = firstnames_df.iloc[:,2:4]
#change the names of the columns so they match the last names columns
firstnames_male.columns=['name','count']
firstnames_female.columns=['name','count']

# pick n_male random male first names, n_female random female first names, and (n_male + n_female) random last names
n_male = 100
n_female = 100
random_lastnames = lastnames_df.sample(n_male + n_female, replace=True, weights='count', axis=0)['name']
random_malenames = firstnames_male.sample(n_male * 2, replace=True, weights='count', axis=0)['name']
random_femalenames = firstnames_female.sample(n_female * 2, replace=True, weights='count', axis=0)['name']

#zip them all together into one big list
full_names = list(zip(random_malenames[:n_male],random_malenames[n_male:2*n_male],random_lastnames[:n_male]))
full_names.extend(list(zip(random_femalenames[:n_female],random_femalenames[n_female:2*n_female],random_lastnames[n_male:])))
full_names = pd.DataFrame(full_names, columns =['First', 'Middle', 'Last'])

#common two-letter abbreviations for people's names
nicknames = ['AJ', 'AW', 'CJ', 'DD', 'DJ', 'ED', 'EJ', 'ET', 'EV', 'GG', 'JB', 'JC', 'JD', 'JJ', \
			 'JK', 'JP', 'JR', 'JT', 'KC', 'KD', 'KJ', 'KP', 'KT', 'LC', 'MJ', 'OJ', 'PJ','RB', \
			 'RJ', 'TC', 'TD', 'TJ', 'TR']

common_middles = ['Marie', 'Ann', 'Anne', 'Lynn', 'Grace', \
			'Rose', 'Jane', 'Louise', 'Jean', \
			'Mae', 'May', 'Lee', 'Michael', 'Paul', 'Joseph', \
			'Robert', 'William', 'Alan', 'David']

full_names['Email'] = ''
full_names['Company'] = ''
full_names['Phone'] = ''

pd.set_option('display.max_rows', None)

#check if it's a common abbreviation
for i in range(len(full_names)):

	initials = full_names['First'][i][0] + full_names['Middle'][i][0]

	full_names['Email'][i] = generate_email(i)

	#get rid of most middle names. For the ones we keep, combine with the first name
	if full_names['Middle'][i] in common_middles:
		full_names['First'][i] = full_names['First'][i] + ' ' + full_names['Middle'][i]
		# print(full_names['Middle'][i])
	full_names['Middle'][i] = ''

	if initials in nicknames:
 		#keep some fraction of these instead of first names
		if np.random.random() < 0.3:
			# Give most of them periods but not all (with variable space between the )
			if np.random.random() < 0.8:
				full_names['First'][i] = initials[0] + '.' + (' ' if np.random.random() < 0.3 else '') + initials[1] + '.'
			else:
				full_names['First'][i] = initials[0] + (' ' if np.random.random() < 0.1 else '') + initials[1]
			# And set the middle name to 0
			full_names['Middle'][i] = ''

	# make some of them backwards
	if np.random.random() < frequencies["f_backwards"]:
		full_names['First'][i] = f"{full_names['Last'][i]}, {full_names['First'][i]}"
		full_names['Last'][i] = ''

	#delete the last name
	if np.random.random() < frequencies["f_first_names_only"]:
		full_names['Last'][i] = ''

	#if you haven't deleted the first name, delete the last name
	if np.random.random() < frequencies["f_last_names_only"] and full_names['Last'][i] != '':
		full_names['First'][i] = ''

	#Delete both names if this is either a company or an email only
	if np.random.random() < frequencies["f_email_only"]:
		full_names['First'][i] = full_names['Email'][i]
		full_names['Last'][i] = ''
	elif np.random.random() < frequencies["f_company_name_only"]:
		full_names['Company'][i] = generate_company()
		full_names['Email'][i] = generate_company_email(full_names['Email'][i],full_names['Company'][i])
		full_names['First'][i] = ''
		full_names['Last'][i] = ''
	elif np.random.random() < frequencies["f_company_plus_name"]:
		full_names['Company'][i] = generate_company()
		full_names['Email'][i] = generate_company_email(full_names['Email'][i],full_names['Company'][i])

	#capitalization
	if np.random.random() < 0.06:
		if np.random.random() < 0.9: full_names['First'][i] = full_names['First'][i].lower()
		if np.random.random() < 0.9: full_names['Last'][i] = full_names['Last'][i].lower()

	full_names['Phone'][i] = generate_phone_number()

	if '@' in full_names['First'][i]:
		full_names['Phone'][i] = ''
	elif np.random.random() < frequencies["f_email_no_number"] - frequencies["f_email_only"]:
		full_names['Phone'][i] = ''
	if np.random.random() < frequencies["f_number_no_email"] and full_names['Phone'][i] != '':
		full_names['Email'][i] = ''

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 500)
print(full_names)