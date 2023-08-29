from faker import Faker
from pprint import pprint

faker = Faker()

n_contacts = 100

profiles = [faker.name() for _ in range(n_contacts)]
phone_numbers = [faker.phone_number() for _ in range(n_contacts)]

pprint(profiles)
pprint(phone_numbers)