import pandas as pd
import re
import datetime

## INPUT FILE PATHS
US_COUNTIES_NPI_CSV_FILE = '../data/complete_npis_inherited_policies_04_22_2020.csv'
US_COUNTIES_CONFIRMED_CASES_CSV_FILE = '../data/us-counties_5_21_2020.csv'
WORLD_NPI_CSV_FILE = '../data/OxCGRT_latest_06_01_2020.csv'


# ========================================================================================== #
# DATA PREPROCESSING FUNCTIONS
# ========================================================================================== #

def get_clean_data(dataset_name="Stanford"):
    if dataset_name == "Stanford":
        return read_preprocess_Stanford_dataset_(US_COUNTIES_NPI_CSV_FILE, US_COUNTIES_CONFIRMED_CASES_CSV_FILE)
    elif dataset_name == "Oxford":
        return read_preprocess_Oxford_dataset_(WORLD_NPI_CSV_FILE)


def read_preprocess_Stanford_dataset_(npi_filepath, confirmed_cases_filepath, debug=False):
    '''
    basically read raw data from file, custom clean it, and put it in proper form for desired NPI analysis.
    returns two data frames: 
        - one data frame containing npi adoption start dates.
        - one containing number of daily confirmed cases.
    '''
    
    ## Read data about NPI start dates in each US county
    
    if debug:
        print('Reading and preprocessing NPI data ...')

    # This data file contains one row per (county, npi) pair
    npi_data_df = pd.read_csv(npi_filepath, delimiter=',', nrows=None)

    if debug:
        nRow, nCol = npi_data_df.shape
        print(f'Original data size: {nRow} rows and {nCol} columns.')

    # remove rows pertaining to states; keep only rows pertaining to counties
    npi_data_df = npi_data_df.loc[npi_data_df.county.notnull() & (npi_data_df.start_date.notnull())]

    # remove rows pertaining to NPI's that were non-adopted by most counties
    idx = npi_data_df.npi.isin(['lockdown', 'Other', 'gathering_size_25_to_11'])
    npi_data_df = npi_data_df[~idx]

    #if debug:
    #    print(f'After filtering irrelevant rows, {npi_data_df.shape[0]} rows remain.')

    # create a combined 'state_county' column in order to uniquely identify counties.
    npi_data_df['county_state'] = npi_data_df.apply(
        lambda row: row.county.strip() + '_' + row.state, axis=1).str.lower()


    # remove irrelevant/useless columns
    keep_columns = ['county_state', 'npi', 'start_date']
    npi_data_df = npi_data_df[keep_columns]
    
    
    # remove invalid dates
    is_valid_date = lambda date: re.match(r'\d{1,2}/\d{1,2}/2020$', date) is not None
    npi_data_df = npi_data_df.loc[npi_data_df.start_date.map(is_valid_date)]

    # convert to datetime object
    npi_data_df['start_date'] = pd.to_datetime(npi_data_df.start_date)

 
    
    ## Read data about number of daily confirmed cases in each US county
    
    if debug:
        print('Reading and preprocessing Confirmed Cases data ...')
  
    # This file contains one row per (county,state,date) tuple
    confirmed_cases_df = pd.read_csv(confirmed_cases_filepath, delimiter=',', nrows=None)

    if debug:
        nRow, nCol = confirmed_cases_df.shape
        print(f'Original data size: {nRow} rows and {nCol} columns.')

    # convert date type from str to datetime object
    confirmed_cases_df['date'] = pd.to_datetime(confirmed_cases_df.date)

    # create state_county column
    confirmed_cases_df['county_state'] = confirmed_cases_df.apply(
        lambda row: row.county.strip() + '_' + row.state, axis=1).str.lower()

    ## verify that (date, county_state) pairs are unique in input data
    u = confirmed_cases_df.groupby(['date', 'county_state']).cases.count()
    assert u.max() == 1 and u.min() == 1

    # remove irrelevant/useless columns
    keep_columns = ['county_state', 'date', 'cases']
    confirmed_cases_df = confirmed_cases_df[keep_columns]

    
    if debug:
        r,c = npi_data_df.shape
        print('Remaining NPI data after cleanup: {r} rows and {c} columns.')
        print('Summary of columns:')
        print(summarize_data(npi_data_df))
        print()

    if debug:
        print('Remaining confirmed cases data after cleanup: {r} rows and {c} columns.')
        print('Summary of columns:')
        print(summarize_data(confirmed_cases_df))
        print()
    
    
    ## Remove counties that are NOT available in both NPI and Confirmed Cases data
    
    u1 = set(confirmed_cases_df.county_state.unique()) 
    u2 = set(npi_data_df.county_state.unique())
    keep_counties = list(u1&u2)

    idx = npi_data_df.county_state.isin(keep_counties)
    npi_data_df = npi_data_df.loc[idx]
    assert set(npi_data_df.county_state.unique().tolist()) == set(keep_counties)

    idx = confirmed_cases_df.county_state.isin(keep_counties)
    confirmed_cases_df = confirmed_cases_df.loc[idx]
    assert set(confirmed_cases_df.county_state.unique().tolist()) == set(keep_counties)

    return npi_data_df, confirmed_cases_df


def read_preprocess_Oxford_dataset_(inputfilepath, debug=False):
    '''
    same interface as read_preprocess_Stanford_dataset_
    '''
    pass

def summarize_data_(df):
    data_summary_df = pd.concat([df.dtypes, df.nunique(), df.notna().sum(), df.isna().sum(), df.apply(
        lambda col: col.unique().tolist() if col.nunique() <= 15 else '', axis=0)], axis=1)
    data_summary_df.columns = ['data type', 'n_distinct_values',
                       'n_available_values', 'n_NaN_values', 'distinct_values']
    return data_summary_df

