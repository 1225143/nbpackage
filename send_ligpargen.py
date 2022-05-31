#!/bin/env python
import os
import sys
import time
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def main(
         smiles='CCC',
         url = 'http://zarbi.chem.yale.edu/ligpargen/'
):

    prefs = {"download.default_directory" : os.getcwd()}
    options = webdriver.ChromeOptions()
#    options.add_argument('--headless')
    options.add_experimental_option('prefs', prefs)
    print(options)

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    time.sleep(10)

    print(driver.title)
    box = driver.find_element_by_id('smiles')
    print(box)
    box.send_keys(smiles)

    time.sleep(10)

    b = driver.find_element_by_class_name("btn")
    print(b)
    b.submit()

    time.sleep(10)

    p = driver.find_elements_by_css_selector('p')
    print(p)
    f = driver.find_element_by_class_name('form-group')
    keys = [_.get_property('value') for _ in  f.find_elements_by_name('go')]
    d = dict(list(zip(keys, f.find_elements_by_name('go'))))
    d['LAMMPS'].submit()

    time.sleep(10)
main(smiles=sys.argv[1])
