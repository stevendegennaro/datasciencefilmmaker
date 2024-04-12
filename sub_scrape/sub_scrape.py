import subprocess
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time # Used for time.sleep()
import os
import datetime as dt
import io
import json
import sys
import traceback

def is_time_between(begin_time:dt.datetime, end_time, check_time):
    '''
    Takes two datetime.time objects and checks if a
    third object falls between them.
    '''
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

def get_current_available_jobs() -> pd.DataFrame:
    '''
    Goes to the substitute assignment website, logs in, 
    and downloads all currently available assinments.
    Returns the result as a pandas DataFrame
    '''
    print("Retrieving current jobs...")
 
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
    available_jobs = pd.read_html(io.StringIO(table_html))[0]
    now = np.datetime64(dt.datetime.today()).astype('datetime64[ns]')

    # Convert dates and times
    available_jobs['Job Start Date'] = pd.to_datetime(available_jobs['Job Start Date'])
    available_jobs['Job End Date'] = pd.to_datetime(available_jobs['Job End Date'])
    available_jobs[['Start Time','End Time']] = available_jobs['Times'].str.strip().str.split("-",expand=True)
    available_jobs["Start Time"] = pd.to_datetime(available_jobs["Start Time"].str.strip(),format = "%I:%M %p")
    available_jobs["Start Time"] = pd.to_numeric(available_jobs["Start Time"].dt.strftime("%H"))
    available_jobs["End Time"] = pd.to_datetime(available_jobs["End Time"].str.strip(),format = "%I:%M %p")
    available_jobs["End Time"] = pd.to_numeric(available_jobs["End Time"].dt.strftime("%H"))
    available_jobs['Download Time'] = now
    available_jobs['Unavailable Time'] = pd.NaT
    available_jobs.drop_duplicates(inplace = True)

    return available_jobs

def archive_jobs_and_return_new(available_jobs: pd.DataFrame) -> pd.DataFrame:
    '''
    Take the list of jobs just downloaded from the website
    and compares it against a list of every job we've already seen.
    Jobs that it hasn't seen before are put into 'new_jobs'.
    If a previously available job is no longer available,
    it records the time and then re-outputs the old jobs list.
    Returns the list of new jobs
    '''
    print("Archiving jobs...")
    previous_open_jobs = pd.read_csv('open_jobs.csv',
                            parse_dates = ['Job Start Date',
                                           'Job End Date',
                                           'Download Time',
                                           'Unavailable Time'])

    # Compare the two lists to see:
    left = previous_open_jobs.drop(['Download Time','Unavailable Time'], axis=1)
    right = available_jobs.drop(['Download Time','Unavailable Time'], axis=1)
    # which jobs that used to be there are now gone
    merged = pd.merge(left,right, how='left', indicator=True).set_axis(previous_open_jobs.index)
    not_open_anymore = merged['_merge'] == 'left_only'
    # which jobs that weren't there before are there now
    merged = pd.merge(left,right, how='right', indicator=True).set_axis(available_jobs.index)
    new_open_jobs = merged['_merge'] == 'right_only'

    # If a job is now gone, set its end time to now and output to the end of the file
    closed_jobs = previous_open_jobs.loc[not_open_anymore]
    closed_jobs.loc[:,'Unavailable Time'] = now
    closed_jobs.to_csv('closed_jobs.csv', mode='a', index=False, header=False)

    # Find the new jobs
    new_jobs = available_jobs[new_open_jobs]

    # Save all open jobs to file
    still_open_jobs = pd.concat([previous_open_jobs.loc[~not_open_anymore],new_jobs])
    still_open_jobs.to_csv('open_jobs.csv',index = False)

    return new_jobs

def get_my_scheduled_jobs()-> list:
    '''
    Calls the Apple Calendar app to get a list
    of days on which I already have a job. 
    Returns a list of dates. If there is an exception
    it returns an empty list
    '''
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
        my_scehduled_jobs = []
        for job in jobs:
            date = job[0]
            while date <= job[1]: 
                my_scehduled_jobs.append(date)
                date += dt.timedelta(days=1)

        my_scehduled_jobs = sorted(list(set(my_scehduled_jobs)))
    except Exception as e:
        print('Error getting calendar events')
        with open("log.txt","a") as f:
            f.write('Error getting calendar events\n')
        my_scehduled_jobs = []
    return my_scehduled_jobs

def filter_jobs(new_jobs:pd.DataFrame) -> pd.DataFrame:
    '''
    Filter the jobs to only the ones that I
    would be interested in taking and return result
    '''

    # If there are no new jobs, there's no reason
    # to check anything else.
    # Just return the empty dataframe
    if len(new_jobs) == 0:
        return new_jobs

    # High schools only
    new_jobs = new_jobs[new_jobs['Role'].str.contains('High')]

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
    filtered_jobs = new_jobs[combined_mask]

    # Remove jobs on days that I already have a job
    if len(filtered_jobs) > 0:
        my_scehduled_jobs = get_my_scheduled_jobs()
        filtered_jobs = filtered_jobs[~filtered_jobs['Job Start Date'].dt.date.isin(my_scehduled_jobs)]

    return filtered_jobs

def create_message(filtered_jobs:pd.DataFrame) -> str:
    '''
    Create a message if there are new jobs.
    If not, print "No new jobs" to screen and log file
    and return None
    '''

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

def send_message(message: str) -> None:
    '''
    Calls applescript to send a message to my phone.
    Global variable 'info' is created in main
    '''
    message = message.replace("'","").replace("\"","")
    message = f"\"{message}\""
    os.system(f"osascript sendMessage.applescript {info['phone_number']} {message}")
    print("Text sent: ",message.strip())
    with open("log.txt","a") as f:
        f.write(f"Text sent: {message.strip()}\n")


##############
###  Main  ###
##############
if __name__ == "__main__":

    # Open the browser in the background
    op = webdriver.ChromeOptions()
    op.add_argument('headless')
    driver = webdriver.Chrome(options=op)

    with open("info.txt","r") as f:
        info = json.load(f)

    last_error = None
    s = 600      # Time to sleep between loops, in seconds
                 # (defaults to 10 minutes, gets changed below
                 # at certain times of day on certain days of the week

    # Main Loop
    while True:


        if os.path.isfile('pause.txt'):
            while os.path.isfile('pause.txt'):
                print('Pausing')
                time.sleep(60)

        if os.path.isfile('stop.txt'):
            print("Stop file found. Exiting...")
            sys.exit()

        now = dt.datetime.now()
        # If it's a weekday evening between 6am and 10pm, or if it's Sunday night between 5 and 10,
        # Check every 2 minutes
        if (now.isoweekday() <= 5 and is_time_between(dt.time(6,0), dt.time(22,0), now.time())) or \
           (now.isoweekday() == 7 and is_time_between(dt.time(15,0), dt.time(22,0), now.time())):
            s = 120

        try:
            available_jobs = get_current_available_jobs()
            new_jobs = archive_jobs_and_return_new(available_jobs)
            filtered_jobs = filter_jobs(new_jobs)
            message = create_message(filtered_jobs)
            if message: send_message(message)

            # If we've made it this far, it means there are no new exceptions
            if last_error:
                send_message(f"Code has resumed at {dt.datetime.now()}")
                last_error = None

        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            with open("log.txt","a") as f:
                f.write(str(tb) + "\n")
            # If this isn't just a repeat of the last exception, then
            if type(e) is not type(last_error) or e.args != last_error.args:
                # send a notification about the exception
                send_message(f"System down at {dt.datetime.now()}\n{type(e).__name__}\n{str(e)[:100]} ")
                last_error = e
                
        if last_error:
            time.sleep(60)
        else:
            time.sleep(s)
