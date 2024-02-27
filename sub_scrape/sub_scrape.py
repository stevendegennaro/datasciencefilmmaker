import subprocess
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import datetime as dt
import io
import json
import sys

def is_time_between(begin_time, end_time, check_time):
    # If check time is not given, default to current UTC time
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

def send_message(message: str):
    message = message.replace("'","").replace("\"","")
    message = f"\"{message}\""
    with open("info.txt","r") as f:
        info = json.load(f)
    os.system(f"osascript sendMessage.applescript {info['phone_number']} {message}")
    print("Text sent: ",message.strip())
    with open("log.txt","a") as f:
        f.write(f"Text sent: {message.strip()}\n")


### Calls the Apple Calendar app to get a list
### of days on which I already have a job. 
### Returns a list of dates. If there is an exception
### it returns an empty list
def get_accepted_jobs():
    print("Checking calendar...")
    applescript_code = '''
    tell application "Calendar"
        set calendarEvents to every event of calendar "Sub"
        set eventsList to {}
        repeat with anEvent in calendarEvents
            set eventStartDate to start date of anEvent as string
            set eventRecurrence to recurrence of anEvent as string
            set eventsList to eventsList & {eventStartDate & "-" & eventRecurrence & "\n"}
        end repeat
        set eventsList to eventsList as string
    end tell
    '''
    try:
        result = subprocess.run(['osascript', '-e', applescript_code], capture_output=True, text=True)
        output = result.stdout.strip()
        jobs = output.split("\n")
        jobs = [job.split("-") for job in jobs]
        date_format = '%A, %B %d, %Y at %I:%M:%S %p'

        for j, job in enumerate(jobs):
            jobs[j][0] = dt.datetime.strptime(job[0], date_format).date()
            if job[1] == 'missing value':
                jobs[j][1] = jobs[j][0]
            else:
                jobs[j][1] = dt.datetime.strptime(job[1].split("UNTIL=")[1][:8],'%Y%m%d').date()
        accepted_dates = []
        for job in jobs:
            date = job[0]
            while date <= job[1]: 
                accepted_dates.append(date)
                date += dt.timedelta(days=1)

        accepted_dates = sorted(list(set(accepted_dates)))
    except Exception as e:
        print('Error getting calendar events')
        with open("log.txt","a") as f:
            f.write('Error getting calendar events\n')
        accepted_dates = []
    return accepted_dates




# This function goes to the substitute assignment website, 
# logs in, and downloads all currently available assinments
# into a Pandas DataFrame. It then removes the assignments
# that don't meet certain criteria. If the list contains
# an assignment that it has not already seen (kept
# in a running list in a file), it sends me a text message.
def get_new_jobs():

    print("Retrieving jobs...")
    with open("info.txt","r") as f:
        info = json.load(f)

    ##################################################################
    ### Get the web page and download the currently available jobs ###
    ##################################################################
 
    # Open the web page
    driver.get(info["url"])

    # Enter ssn and password and click Sign On button
    ssn_element = driver.find_element(By.ID,"last4ofSSN")
    ssn_element.send_keys(info["last4ofSSN"])
    pin_element = driver.find_element(By.ID,"pin")
    pin_element.send_keys(info["pin"])
    button = driver.find_element(By.ID,"ok")
    button.click()

    # Wait for the page to load, then click "Search for Jobs"
    wait = WebDriverWait(driver, 10)
    link = wait.until(EC.presence_of_element_located((By.ID, "subNav-jobSearch")))
    link.click()

    # Wait for the table of available jobs to load, then put into a data frame
    wait = WebDriverWait(driver, 10)
    table = wait.until(EC.presence_of_element_located((By.ID, "tableTable")))
    table_html = table.get_attribute("outerHTML")
    jobs = pd.read_html(io.StringIO(table_html))[0]
    now = np.datetime64(dt.datetime.today()).astype('datetime64[ns]')

    # Convert dates and times
    jobs['Job Start Date'] = pd.to_datetime(jobs['Job Start Date'])
    jobs['Job End Date'] = pd.to_datetime(jobs['Job End Date'])
    jobs[['Start Time','End Time']] = jobs['Times'].str.strip().str.split("-",expand=True)
    jobs["Start Time"] = pd.to_datetime(jobs["Start Time"].str.strip(),format = "%I:%M %p")
    jobs["Start Time"] = pd.to_numeric(jobs["Start Time"].dt.strftime("%H"))
    jobs["End Time"] = pd.to_datetime(jobs["End Time"].str.strip(),format = "%I:%M %p")
    jobs["End Time"] = pd.to_numeric(jobs["End Time"].dt.strftime("%H"))
    jobs['Download Time'] = now
    jobs['Unavailable Time'] = pd.NaT

    # High schools only
    jobs = jobs[jobs['Role'].str.contains('High')]

    #################################################################
    ### Take the list of jobs just downloaded                     ###
    ### from the website and compare it against a list of         ###
    ### every job we've already seen. Jobs that it hasn't seen    ###
    ### before are put into 'new_jobs'. If a previously available ###
    ### job is no longer available, it records the time and then  ###
    ### re-outputs the old jobs list.                             ###
    #################################################################
    old_jobs = pd.read_csv('old_jobs.csv',
                            parse_dates = ['Job Start Date',
                                           'Job End Date',
                                           'Download Time',
                                           'Unavailable Time'])

    # Get rid of old_jobs that we've already recorded an Unavailable Time for
    check_jobs = old_jobs[old_jobs['Unavailable Time'].isnull()]
    # Compare the two lists to see:
    left = check_jobs.drop(['Download Time','Unavailable Time'], axis=1)
    right = jobs.drop(['Download Time','Unavailable Time'], axis=1)
    # which jobs that used to be there are now gone
    unavailable = left.merge(right, how='left', indicator=True).set_axis(check_jobs.index)
    unavailable = unavailable['_merge'] == 'left_only'
    # which jobs that weren't there before are there now
    new = left.merge(right, how='right', indicator=True).set_axis(jobs.index)
    new = new['_merge'] == 'right_only'

    # If a job is now gone, set its end time to now
    check_jobs.loc[unavailable,'Unavailable Time'] = now
    # Find the new jobs
    new_jobs = jobs[new]
    # Add the new jobs to the old jobs and jobs that are newly unavailable
    old_jobs = old_jobs[~old_jobs['Unavailable Time'].isnull()] 
    old_jobs = pd.concat([old_jobs,check_jobs,new_jobs])
    # Output the result
    old_jobs.to_csv('old_jobs.csv',index = False)

    ###############################################
    ### Filter the jobs to only the ones that I ###
    ### would be interested in taking           ###
    ###############################################

    # Always include half-day assignments on M, W, F (0,2,4)
    half_day = (
                    (new_jobs['Job Start Date'].dt.dayofweek.isin([0,2,4])) & 
                    (new_jobs['Day Count'] == 1) & 
                    (new_jobs["End Time"] < 14)
               )

    # Don't inlcude jobs that are only afternoon jobs
    afternoon_only = new_jobs["Start Time"] > 10

    # Don't include all day jobs on M and W
    mw_all_day = (
                    (new_jobs['Job Start Date'].dt.dayofweek.isin([0,2])) & 
                    (new_jobs["End Time"] > 14)
                 )

    # Only at certain schools, 
    hs_list = info["hs_list"]
    hs_masks = [new_jobs['Organization'].str.contains(o) for o in hs_list]

    # Put all the masks together
    combined_mask = (
                        half_day | 
                        (
                            pd.concat(hs_masks, axis=1).any(axis=1) & 
                            (~afternoon_only) &
                            (~mw_all_day) &
                            # Only include jobs that are 2 days or fewer
                            (new_jobs['Day Count'] <= 2) &
                            # Only include jobs that are in the next 30 days
                            ((new_jobs["Job Start Date"] - dt.datetime.today()).dt.days < 30)
                        )
                    )
    filtered_jobs = new_jobs[combined_mask].copy()

    ############################
    ### Calendar integration ###
    ############################

    # Remove jobs on days I already have a job
    accepted_jobs = get_accepted_jobs()
    filtered_jobs = filtered_jobs[~filtered_jobs['Job Start Date'].dt.date.isin(accepted_jobs)]

    ###########################################
    ### Create messge if there are new jobs ###
    ###########################################

    if len(filtered_jobs) > 0:
        # If there are new jobs, send me a text
        # Include date, day of the week, number of days, school, time, and Job Title
        message = f"{len(filtered_jobs)} new job{'s' if len(filtered_jobs) > 1 else ''}:\n"
        for _,row in filtered_jobs.iterrows():
            message += (f"{row['Job Start Date'].strftime('%A, %m/%d')}, " +
                        f"{row['Day Count']} day(s), " +
                        f"{row['Times']}, " +
                        f"{row['Organization']}, " +
                        f"{row['Job Title']}, " +
                        f"{row['Employee']}\n\n")

    else:
        print('No new jobs',dt.datetime.now())
        with open("log.txt","a") as f:
            f.write(f'No new jobs {dt.datetime.now()}\n')
        message = None

    return message

##############
###  Main  ###
##############

# Open the browser in the background
op = webdriver.ChromeOptions()
op.add_argument('headless')
driver = webdriver.Chrome(options=op)

last_error = None
s = 600      # Time to sleep between loops, in seconds
             # (defaults to 10 minutes, gets changed below
             # at certain times of day on certain days of the week

# Main Loop
while True:

    if os.path.isfile('pause.txt'):
        while os.path.isfile('pause.txt'):
            time.sleep(60)

    if os.path.isfile('stop.txt'):
        print("Stop file found. Exiting...")
        sys.exit()

    now = dt.datetime.now()
    # If it's a weekday evening between 6am and 10pm, or if it's Sunday night between 5 and 10,
    # Check every 2 minutes
    if now.isoweekday() <= 5 and is_time_between(dt.time(6,0), dt.time(22,0), now.time()) | \
       now.isoweekday() == 7 and is_time_between(dt.time(15,0), dt.time(22,0), now.time()):
        s = 120

    try:
        message = get_new_jobs()
        # if message: send_message(message)
        if message: print(message)

        # If we've made it this far, it means there are no new exceptions
        if last_error:
            send_message(f"Code has resumed at {dt.datetime.now()}")
            last_error = None

    except Exception as e:
        print(e)
        with open("log.txt","a") as f:
            f.write(str(e)+"\n")
        # If this isn't just a repeat of the last exception, then
        if type(e) is not type(last_error) or e.args != last_error.args:
            # send a notification about the exception
            send_message(f"System down with error {str(e)[:100]} at {dt.datetime.now()}")
            last_error = e
            
    if last_error:
        time.sleep(60)
    else:
        time.sleep(s)
