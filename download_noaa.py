import os
import requests
import pandas as pd


#Accesses the website
TOKEN = "oXHZIzVZSfbaWZFGRCakhivwOhtpnHJZ"    
headers = {"token": TOKEN}
base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"


#Loops over all the relevant dates and writes it to a csv file
for i in range(10,31,2):
    print(i)
    params = {
        "datasetid": "GHCND",
        "datatypeid": "PRCP",
        "startdate": f"2019-04-{i}",
        "enddate":   f"2019-04-{i+1}",
        "locationid": "FIPS:42",       # Pennsylvania FIPS code = 42
        "limit": 1000,
        "units": "metric",             # mm
        "includemetadata": "false"
    }

    # fetch data
    response = requests.get(base_url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json().get("results", [])

    # ── load into pandas dataframe
    df = pd.DataFrame(data)
    # Columns include: station, date, datatype (PRCP), and value (hundredths of mm)
    df = df.assign(
        date=pd.to_datetime(df["date"]).dt.date,
        precip_mm=lambda d: d["value"] / 10  # convert from tenths of mm to mm
    )[["date","station","precip_mm"]]

    print(df.head())
    df.to_csv('data.csv', mode='a', index=False, header=False)


