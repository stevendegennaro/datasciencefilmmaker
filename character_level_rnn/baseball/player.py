class Player:
    def __init__(self,entry: str) -> None:
        years = entry[-13:]
        if years[0] == '+':
            self.hof = True
        else:
            self.hof = False
        self.start_year = years[3:7]
        self.end_year = years[8:12]

        fullname = entry[:-13].split(" ")
        
        #check for a suffix
        if (fullname[-1] in ['Jr.','Sr.','II','III','IV']):
            self.suffix = fullname[-1]
            fullname = fullname[:-1]
        else:
            self.suffix = None

        #check if they only have one name (in which case it is a last name)
        if len(fullname) == 1:
            self.lastname = fullname[0]
            self.firstname = None
        else:
            #check if they have more than two names
            if len(fullname) > 2:
                #if it's one of these cases, combine the last names
                if 'de' in fullname:
                    if fullname[1] in ['Montes','Ponce']:
                        self.firstname = fullname[0]
                        self.lastname = " ".join(fullname[1:4])
                    else:
                        i = fullname.index('de')
                        self.firstname = " ".join(fullname[:i])
                        self.lastname = " ".join(fullname[i:])
                elif 'De' in fullname:
                        i = fullname.index('De')
                        self.firstname = " ".join(fullname[:i])
                        self.lastname = " ".join(fullname[i:])
                elif fullname[-2] in ['Dal','Del','den','Des','La','Lo','Santo','St.','Van','Vande','Vander','Von','Yellow','Woods']:
                        self.firstname = fullname[0]
                        self.lastname = " ".join(fullname[1:])
                #otherwise, combine the first names
                else:
                    self.firstname = " ".join(fullname[:-1])
                    self.lastname = fullname[-1]
            #the rest all have exactly two names: first name and last name
            else:
                self.firstname = fullname[0]
                self.lastname = fullname[1]