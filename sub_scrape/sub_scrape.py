#ADD:
# Delete old jobs that aren't in new jobs
# Calendar integration

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

# This function oes to the substitute assignment website, 
# logs in, and downloads all currently available assinments
# into a Pandas DataFrame. It then removes the assignments
# that don't meet certain criteria. If the list contains
# an assignment that it has not already seen (kept
# in a running list in a file), it sends me a text message.
def get_new_jobs():

    print("Retreiving jobs...")
    with open("info.txt","r") as f:
        info = json.load(f)

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

    # Convert dates and times
    jobs['Job Start Date'] = pd.to_datetime(jobs['Job Start Date'])
    jobs['Job End Date'] = pd.to_datetime(jobs['Job End Date'])
    jobs[['Start Time','End Time']] = jobs['Times'].str.strip().str.split("-",expand=True)
    jobs["Start Time"] = pd.to_datetime(jobs["Start Time"].str.strip(),format = "%I:%M %p")
    jobs["Start Time"] = pd.to_numeric(jobs["Start Time"].dt.strftime("%H"))
    jobs["End Time"] = pd.to_datetime(jobs["End Time"].str.strip(),format = "%I:%M %p")
    jobs["End Time"] = pd.to_numeric(jobs["End Time"].dt.strftime("%H"))

    # High schools only
    jobs = jobs[jobs['Role'].str.contains('High')]

    # Always include half-day assignments on M, W, F (0,2,4)
    half_day = (
                    (jobs['Job Start Date'].dt.dayofweek.isin([0,2,4])) & 
                    (jobs['Day Count'] == 1) & 
                    (jobs["End Time"] < 14)
               )

    # Don't inlcude jobs that are only afternoon jobs
    afternoon_only = jobs["End Time"]
    # Don't include all day jobs on M and W
    mw_all_day = (
                    (jobs['Job Start Date'].dt.dayofweek.isin([0,2])) & 
                    (jobs["End Time"] > 14)
                 )

    # Only at certain schools, 
    hs_list = info["hs_list"]
    hs_masks = [jobs['Organization'].str.contains(o) for o in hs_list]

    # Put all the masks together
    combined_mask = (
                        half_day | 
                        (
                            pd.concat(hs_masks, axis=1).any(axis=1) & 
                            (~afternoon_only) &
                            (~mw_all_day) &
                            # Only include jobs that are 2 days or fewer
                            (jobs['Day Count'] <= 2) &
                            # Only include jobs that are in the next 30 days
                            ((jobs["Job Start Date"] - dt.datetime.today()).dt.days < 30)
                        )
                    )
    filtered_jobs = jobs[combined_mask].copy()
    filtered_jobs = filtered_jobs.drop(['Start Time','End Time'],axis=1)
    filtered_jobs.to_csv('filtered_jobs.csv', index=False)

    # Check if we already have an old jobs file
    if os.path.isfile('old_jobs.csv'):
        # Read it in
        old_jobs = pd.read_csv('old_jobs.csv',parse_dates = ['Job Start Date','Job End Date'])
        if not old_jobs.empty:
            # Get rid of anything that ended before today
            old_jobs = old_jobs[old_jobs['Job End Date'] >= pd.to_datetime(dt.datetime.now().date())]
            # Get rid of anything that's not in our new list
            # (i.e., the job has been deleted or claimed by someone else)
            old_jobs = old_jobs.merge(filtered_jobs, how='left', indicator=True)
            old_jobs = old_jobs[old_jobs['_merge'] == 'both']
            old_jobs.drop(columns='_merge',inplace = True)
        else:
            # Otherwise, create an empty data frame
            old_jobs = pd.DataFrame(columns = filtered_jobs.columns).astype(filtered_jobs.dtypes.to_dict())

    else:
        # Otherwise, create an empty data frame
        old_jobs = pd.DataFrame(columns = filtered_jobs.columns).astype(filtered_jobs.dtypes.to_dict())

    # Check file for each job to see if we've already seen it
    new_jobs = filtered_jobs[~(filtered_jobs['Employee'].isin(old_jobs['Employee']) & \
                             filtered_jobs['Job Start Date'].isin(old_jobs['Job Start Date']))]

    # Remove jobs on days I already have a job
    accepted_jobs = get_accepted_jobs()
    new_jobs = new_jobs[~new_jobs['Job Start Date'].dt.date.isin(accepted_jobs)]

    # Output jobs list for comparison next time through the loop
    old_jobs = pd.concat([old_jobs,new_jobs])
    old_jobs.to_csv('old_jobs.csv',index=False)

    if len(new_jobs) > 0:
        # If there are new jobs, send me a text
        # Include date, day of the week, number of days, school, time, and Job Title
        message = f"{len(new_jobs)} new job{'s' if len(new_jobs) > 1 else ''}:\n"
        for _,row in new_jobs.iterrows():
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
    now = dt.datetime.now()
    # If it's a weekday evening between 6am and 10pm, or if it's Sunday night between 5 and 10,
    # Check every 2 minutes
    if now.isoweekday() <= 5 and is_time_between(dt.time(6,0), dt.time(22,0), now.time()) | \
       now.isoweekday() == 7 and is_time_between(dt.time(15,0), dt.time(22,0), now.time()):
        s = 120

    try:
        message = get_new_jobs()
        if message: send_message(message)

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
