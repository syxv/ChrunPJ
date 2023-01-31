from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from flask import request

from datetime import datetime, timedelta
import time
import random
import os
import json


head = "https://www.indeed.com/"

chromedriver = "/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver

job_titles = ["data+scientist"]

def get_soup(url):
    """
    This function get the beautifulsoup object of a webpage.

    Args:
        url (str): the link string of webpage

    Returns:
        soup (obj): beautifulsoup object
    """
    request = request(url, headers={'User-Agent': 'Resistance is futile'})
    response = urlopen(request)
    return BeautifulSoup(response, "html.parser")

def get_jobs_of_title(job_title):
    """
    Args:
        job_title (str): example: 'data+scientist'

    Returns:
    """

    #needed to be changed
    num_pages = 1 #number of pages to scrape
    page_gap_min = 3 #min sleep time between pages
    page_gap_max = 5 #max sleep time between pages
    job_per_page = 50 #number of jobs in one page
    job_gap_min = 5 #min sleep time between jobs
    job_gap_max = 6 #max sleep time between jobs

    for i in range(num_pages): 
        #sleep between each call
        gap = random.uniform(page_gap_min,page_gap_max) 
        time.sleep(gap)

        #each page contains 50 jobs
        tail = "jobs?q={0}&sort=date&limit={1}".format(job_title,job_per_page)
        if i>0:
            tail += "&start={0}".format(i*job_per_page)

        #get link to joblist page
        url = head+tail 
         
        #get links to webpages of jobs on the joblist
        job_page_links = get_job_links_from_page(url)

        for job_page_link in job_page_links:
            gap = random.uniform(job_gap_min,job_gap_max) 
            time.sleep(gap)
            data = get_info_from_job_page(job_page_link)

            print(json.dumps(data))

def get_job_links_from_page(url):
    """
    This function gets the links of the jobs on the joblist page.

    Args:
        url (str): link to joblist page

    Returns:
        job_page_links (list): list of links to the webpages of the jobs
    """

    job_page_links = []
    soup = get_soup(url)
    for item in soup.find_all("a", href=True):
        if '/rc/clk?jk=' in str(item) and 'fccid=' in str(item):
            link = item['href'].split("clk?")[1]
            job_page_links.append(head+'viewjob?'+link)
    return job_page_links

def get_info_from_job_page(url):
    """
    This function get all the useful info from the job webpage.

    Args:
        url (str): link to job webpage

    Returns:
        data (dict): dictionary with keywords: 
                     time_stamp, original_link, job_title, location, company, description
    """
    soup = get_soup(url)
    data = {}
    time_str = soup.find('div',class_='result-link-bar').find('span').getText()

    try:
        data["time_stamp"] = get_timestamp(time_str).strftime("%d-%m-%Y %H:%M")
        data["job_title"] = soup.find('b', class_='jobtitle').getText()
        data["location"] = soup.find('span', class_='location').getText()
        data["company"] = soup.find('span', class_='company').getText()
        data["description"] = soup.find('td',class_='snip').find('div').getText()

        re_link = soup.find('a',class_='sl ws_label')['href'].split("&from=")[0]
        re_link = head[:-1]+re_link
        data["original_link"] = get_original_link(re_link)
    except:
        pass
    return data

def get_timestamp(time_str):
    """
    Calculate the timestamp from the time string.
    
    Args:
        time_str (str): time string, like '2 hours ago'

    Returns:
        time_stamp (obj): timestamp object
    """
    if 'hour' in time_str:
        lag = int(time_str.split('hour')[0])
        delta = timedelta(hours=lag)
        now = datetime.utcnow().replace(second=0,minute=0)
        return now-delta
    else:
        return -1

def get_original_link(url):
    """
    Get the original link of the job description.
    
    Args:
        url (str): the link in Indeed database

    Returns:
        url (str): the original link to the job description
    """
    driver = webdriver.Chrome(chromedriver)
    driver.get(url)
    time.sleep(2)
    original_url = driver.current_url
    driver.quit()
    return original_url


if __name__ == "__main__":
    get_jobs_of_title("data+scientist")
