import vcf_creator
import pandas as pd

def import_contacts():
    contacts = pd.read_csv('data/new_contacts.csv')
    return contacts