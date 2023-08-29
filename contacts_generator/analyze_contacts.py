import pandas as pd
import re
from pprint import pprint

filename = 'data/Contacts/Contacts.vcf'
contacts_df = pd.read_csv('data/Contacts/Contacts.csv',low_memory=False)
n_contacts = len(contacts_df.index)

names_df = contacts_df.loc[:,['FirstName', 'MiddleName', 'LastName', 'Nickname',
       'Company']]

#if there's a middle name and no last name, move the middle name to the last name
names_df.loc[(names_df["MiddleName"].notna()) & 
		     (names_df["LastName"].isna()),"LastName"] = names_df.loc[(names_df["MiddleName"].notna()) & 
		     (names_df["LastName"].isna()),"MiddleName"]

#calculate various stats
first_names_only = names_df.loc[(names_df["LastName"].isna()) & (names_df["Company"].isna())]
n_first_names_only = len(first_names_only.loc[~(first_names_only["FirstName"].str.contains('@').fillna(False)) & 
						   ~(first_names_only["FirstName"].str.contains(' ').fillna(False))])

last_names_only = names_df.loc[(names_df["FirstName"].isna()) & (names_df["Company"].isna())]
n_last_names_only = len(last_names_only.loc[~(last_names_only["LastName"].str.contains('@').fillna(False)) & 
						   ~(last_names_only["LastName"].str.contains(' ').fillna(False))])

n_company_name_only = len(names_df.loc[(names_df["LastName"].isna()) 
		& (names_df["Company"].notna()) 
		& (names_df["FirstName"].isna())])

n_email_only = len(names_df.loc[(names_df["LastName"].isna()) &
					(names_df["FirstName"].str.contains('@').fillna(False))])  + \
			   len(names_df.loc[(names_df["FirstName"].isna()) &
					(names_df["LastName"].str.contains('@').fillna(False))]) - 16 #There are 16 duplicates

n_backwards = len(first_names_only.loc[~(first_names_only["FirstName"].str.contains('@').fillna(False)) &
							 (first_names_only["FirstName"].str.contains("\,").fillna(True))])

n_initials = len(names_df.loc[~(names_df["FirstName"].str.contains('@').fillna(True)) &
							  ~(names_df["LastName"].str.contains('@').fillna(False)) &
							   (names_df["FirstName"].str.contains("\.").fillna(True))])

n_company_plus_name = len(names_df.loc[names_df["Company"].notna() & names_df["FirstName"].notna()])


def get_frequencies():
	frequencies = {
		'f_company_name_only': n_company_name_only/n_contacts,
		'f_company_plus_name': n_company_plus_name/n_contacts,
		'f_first_names_only': n_first_names_only/n_contacts,
		'f_last_names_only': n_last_names_only/n_contacts,
		'f_email_only': n_email_only/n_contacts,
		'f_backwards': n_backwards/n_contacts,
		'f_initials': n_initials/n_contacts,
		'f_email_no_number': n_email_no_number/n_contacts,
		'f_number_no_email': n_number_no_email/n_contacts
	}
	return frequencies

#Combine all the emails into one big set
email_df = contacts_df[[col for col in contacts_df.columns if 'Email' in col]]
emails = [x for col in email_df.columns for x in email_df[col] if (pd.notna(x) and '@' in str(x))]
emails = list(set(emails)) # remove duplicates

#Combine all phone numbers into one big list
numbers_df = contacts_df[[col for col in contacts_df.columns if 'Phone' in col]]
numbers = [x for col in numbers_df.columns for x in numbers_df[col] if (pd.notna(x))]
number_list = []

has_email = email_df.notna().any(axis=1)
has_number = numbers_df.notna().any(axis=1)
n_email_no_number = (has_email & ~has_number).sum()
n_number_no_email = (~has_email & has_number).sum()

#Clean up the numbers
for number in numbers:
	res = re.findall(r'"(.*?)"', number)[0].replace(u'\xa0', u' ')
	res = re.sub(r"\W+", "",res)
	#remove country code
	if res[0] == '1':
		res = res[1:]
	#don't include international numbers
	if len(res) == 10:
		number_list.append(res)
number_list = list(set(number_list)) #remove duplicates
area_codes = [number[0:3] for number in number_list]


def get_emails():
	return emails

def get_area_codes():
	return area_codes

