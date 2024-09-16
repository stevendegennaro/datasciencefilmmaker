import pandas as pd

def import_contacts():
    contacts = pd.read_csv('data/new_contacts.csv',index_col=0)
    contacts.fillna("",inplace=True)
    return contacts

def create_vcf(contacts:pd.DataFrame,filename:str):
    contacts.fillna('',inplace=True)
    # contacts['N'] = contacts['Last'] + ';' + contacts['First'] + ';' + contacts['Middle']
    contacts['VCard'] = 'BEGIN:VCARD\nVERSION:4.0\nFN:\nN:' + \
        contacts['Last'] + ';' + contacts['First'] + ';' + contacts['Middle'] + '\n' + \
        'ORG:' + contacts['Company'] + '\n' + \
        'TEL:' + contacts['Phone'] + '\n' + \
        'EMAIL:' + contacts['Email'] + '\n' + \
        'END:VCARD\n'

    with open(filename,'w') as f:
        for index,row in contacts.iterrows():
            # print(row['VCard'])
            f.write(row['VCard'])
    # return contacts['VCard']