''' The preprocessing module reads the dataset and puts it into a common form for both datsets. '''

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

def get_clean_data(dataset_name):
    '''
    Returns two data frames: 
        - one containing npi adoption start dates, of the form: location, npi, start_date
        - one containing daily cumulative number of confirmed cases, of the form: location, date, cases
    '''

    if dataset_name == "Stanford":
        return read_preprocess_Stanford_dataset_(US_COUNTIES_NPI_CSV_FILE, US_COUNTIES_CONFIRMED_CASES_CSV_FILE)
    elif dataset_name == "Oxford":
        return read_preprocess_Oxford_dataset_(WORLD_NPI_CSV_FILE)


def show_raw_data(dataset_name, n=10):
    npi_filepath = US_COUNTIES_NPI_CSV_FILE if dataset_name == "Stanford" else WORLD_NPI_CSV_FILE
    data_df = pd.read_csv(npi_filepath, delimiter=',', nrows=None)
    nrows, ncols = data_df.shape
    print("Raw dataset file contains {nrows} rows and {ncols} columns.")
    if n == -1:
        return data_df
    else:
        return data_df.head(n)

def read_preprocess_Stanford_dataset_(npi_filepath, confirmed_cases_filepath, debug=False):
    '''
    Read and preprocess Stanford dataset files.
    Returns two separate data frames, one for NPI data and one for confirmed cases.
    '''


    ## Read NPI data file of US states and counties
    # Each row of this data file contains data of a unique (us_county, npi) pair
    # The data in each row consists of the start date of a particular NPI in a particular US county
    
    if debug:
        print('Reading and preprocessing NPI data ...')

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

    # convert date column to datetime type
    npi_data_df['start_date'] = pd.to_datetime(npi_data_df.start_date)

 
    
    ## Read data about number of daily confirmed cases in each US county
    # Each row of this data file contains the cumulative number of confirmed cases of a unique (us_county, date) pair

    if debug:
        print('Reading and preprocessing Confirmed Cases data ...')
  
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
    Read and preprocess the Oxford dataset file.
    Returns two separate data frames, one for NPI data and one for confirmed cases.
    '''

    ## Read raw data file
    # Each row of this file contains data for a unique (countRy, date) pair ;
    #And the data in each row consists of the current strictness level of every NPI for a particular countRy on a particular date,
    #  as well as the cumulative number of confirmed Covid19 cases in that country on that date.
    #  The strictness level of an NPI is an integer in the range 0-4, a value 0 indicates the NPI has not been adopted.

    raw_data_df = pd.read_csv(inputfilepath, delimiter=',', nrows=None)

    if debug:
        nRow, nCol = raw_data_df.shape
        print(f'Original data size: {nRow} rows and {nCol} columns.')

    ## Remove irrelevant/useless columns
    # will keep only NPI columns, Date column, and confirmed cases column
    columns_to_remove = ['CountryCode', 'ConfirmedDeaths', 'M1_Wildcard', \
                         'H4_Emergency investment in healthcare', 'H5_Investment in vaccines' ]
    columns_to_remove += raw_data_df.columns[raw_data_df.columns.str.contains('Index')]
    columns_to_remove += raw_data_df.columns[raw_data_df.columns.str.startswith('E') | raw_data_df.columns.str.endswith('_Flag') ].tolist()

    raw_data_df.drop(columns_to_remove, inplace=True, axis=1)

    ## Mark the columns corresponding to NPI's for later use
    NPI_columns = sorted(list(set(raw_data_df.columns) - {'CountryName', 'Date', 'ConfirmedCases'}))

    
    ## Convert date column to type datetime object
    raw_data_df['Date'] = pd.to_datetime(raw_data_df.Date, yearfirst=True, format="%Y%m%d")

    ## Remove countries with too few cases
    NumCases_Threshold = 200
    u = raw_data_df.groupby('CountryName')['ConfirmedCases'].agg(max)
    countries_to_remove = u[(u < NumCases_Threshold) | u.isna()].index.tolist()
    idx = raw_data_df.CountryName.isin(countries_to_remove)
    raw_data_df.drop(raw_data_df.CountryName[idx].index, axis=0, inplace=True)


    ## Extract confirmed cases data frame

    confirmed_cases_df = raw_data_df[['CountryName','Date','ConfirmedCases']]

    def fill_missing_values_(ts):
        ## fill-in leading missing values with 0
        i = ts.notnull().index.min()
        assert all(ts[ts.index<i].isnull() | (ts[ts.index<i]==0))
        ts.loc[ts.index<i] = 0
        ## forward fill trailing missing values
        ts.fillna(method='ffill',inplace=True)
        assert all(ts.notna())

    for country in confirmed_cases_df.CountryName.unique():
        ts = confirmed_cases_df.loc[confirmed_cases_df.CountryName == country].set_index('Date')['ConfirmedCases']
        fill_missing_values_(ts)



    ## Extract NPI data frame
    
    import itertools
    L = []
    
    for country, npi in itertools.product(df.CountryName.unique(), NPI_columns):
        npi_ts = df1.loc[df1.CountryName==country,['Date', npi]].set_index('Date')
        # cleanup time series
        # remove missing values
        # remove rows with zero value

        start_dates = npi_ts.reset_index().pivot_table(columns=npi, values='Date', aggfunc=min).T.reset_index()
        assert start_dates.shape[1] <= df1[npi].nunique()  # number of strictness levels
        assert start_dates.shape[1] == 2
        start_dates[npi] = start_dates[npi].map(lambda x: f'{npi}_{int(x)}')
        L.append([(country,x[0],x[1]) for i,x in start_dates.iterrows()])

    npi_data_df = pd.DataFrame(L)

    ## iterate between removing rows and columns containing missing values
    # until there are no missing values
    #   1. remove NPI's (columns) that contain too many nan values
    #   2. remove countries (rows) that contain too many nan values
    #   3. go back to 1

    return npi_data_df, confirmed_cases_df


# ------------------------------------ UNUSED FUNCTIONS ------------------------------------- #

def summarize_data_(df):
    data_summary_df = pd.concat([df.dtypes, df.nunique(), df.notna().sum(), df.isna().sum(), df.apply(
        lambda col: col.unique().tolist() if col.nunique() <= 15 else '', axis=0)], axis=1)
    data_summary_df.columns = ['data type', 'n_distinct_values',
                       'n_available_values', 'n_NaN_values', 'distinct_values']
    return data_summary_df

