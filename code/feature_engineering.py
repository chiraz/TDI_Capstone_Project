import pandas as pd
import datetime
import math

# ========================================================================================== #
# FEATURE ENGINEERING FUNCTIONS
# ========================================================================================== #

def date_Nth_case(confirmed_cases_df, N):
    ''' returns date of Nth case for a given value of N'''
    return get_confirmed_cases_time_series_(confirmed_cases_df).apply(lambda x: x[x>=N].index.min(), axis=0)

def feature_engineering(npi_data_df, confirmed_cases_df, N=100, D=14, num_periods=1, smoothing_ws=3):
    '''
      
      : number of cases that characterize the onset of the disease in a country or county; used for normalization purposes.
      D: number of days after Nth case that we observe the target infection rate.
      num_periods : interval size in number of days for calculating infection rate :
           where InfectionRate(t) = (ConfirmedCases(t)-ConfirmedCases(t-num_periods)) / ConfirmedCases(t-num_periods)
      smoothing_ws : window size for smoothing infection rate time series.
    '''

    ## extract time series of confirmed cases for each location
    confirmed_cases_ts = get_confirmed_cases_time_series_(confirmed_cases_df)

    ## Calculate infection rate time series for each location
    infection_rate_ts = calc_infection_rate_time_series_(confirmed_cases_df, num_periods, smoothing_ws)

    ## calculate date of Nth case for each location
    reference_date0 = confirmed_cases_ts.apply(lambda x: x[x>=N].index.min(), axis=0)   # date of Nth case
    
    ## align each time series by date of Nth case
    aligned_infection_rate_ts = align_time_series_( infection_rate_ts, reference_date0)

    ## calculate normalized NPI matrix
    reference_date1 = reference_date0 + datetime.timedelta(days=D)   # date of Nth case + D days
    npi_aligned_matrix = extract_NPI_features_(npi_data_df, reference_date1)

    return npi_aligned_matrix, aligned_infection_rate_ts.T


def get_confirmed_cases_time_series_(confirmed_cases_df):
    '''
      Input: confirmed_cases_df : data frame confirmed_cases containing rows of the form: location, date, cases
      Output: time series of confirmed cases for each location
    '''

    colnames = confirmed_cases_df.columns
    assert len(colnames) == 3
    location_col, date_col, cases_col = colnames

    ## extract non-aligned confirmed cases matrix; the columns in this matrix correspond to real start dates.
    #confirmed_cases_ts = confirmed_cases_df.pivot(index='date', columns='county_state', values='cases')
    confirmed_cases_ts = confirmed_cases_df.pivot(index=date_col, columns=location_col, values=cases_col)

    ## fill-in leading missing values with 0's in each time series
    # because they indicate 0 confirmed cases at that date.

    confirmed_cases_ts.fillna(value=0, inplace=True)
    assert confirmed_cases_ts.isnull().sum().sum() == 0

    return confirmed_cases_ts


def get_new_confirmed_cases_(confirmed_cases_df, smoothing_ws=3, N=None):
    confirmed_cases_ts = get_confirmed_cases_time_series_(confirmed_cases_df)
    new_cases_ts = confirmed_cases_ts.diff(periods=1, axis=0).rolling(window=smoothing_ws).mean()

    # fill-in leading missing values; they are just an artifact of differencing.
    new_cases_ts.fillna(method='backfill', inplace=True)
    assert new_cases_ts.isnull().sum().sum() == 0

    if N is None:
        return new_cases_ts
    else:
        reference_date = confirmed_cases_ts.apply(lambda x: x[x>=N].index.min(), axis=0)  # date of Nth case
        return align_time_series_(new_cases_ts, reference_date)


def calc_infection_rate_time_series_(confirmed_cases_df, num_periods, smoothing_ws):

    confirmed_cases_ts = get_confirmed_cases_time_series_(confirmed_cases_df)
    calc_change_rate = lambda x, n, w: x.fillna(method='ffill') \
             .pct_change(periods=n, fill_method='ffill') \
             .rolling(window=w).mean()  ##.median()

    return confirmed_cases_ts.apply(lambda x: calc_change_rate(x, n=num_periods, w=smoothing_ws), axis=0)



def align_time_series_(ts, reference_date):
    '''
    Input: 

    Output: two matrix: where rows represent geographic units, columns represent number of 
                days since the Nth confirmed case in a geographic unit, and values
                represent the number of confirmed cases, and the infection rate, respectively.
    '''

    extract_ts = lambda x,date:  x.loc[x.index>=date].reset_index(drop=True)

    d = dict()
    for colname,coldata in ts.iteritems():
        d[colname] = extract_ts(coldata,reference_date[colname])
    aligned_ts = pd.DataFrame(d)
    aligned_ts.index.name = 'n_days'
    aligned_ts.columns.name = ts.columns.name
    
    return aligned_ts


def extract_NPI_features_(npi_data_df, reference_date):
    '''
    Input: data frame npi_data containing rows of the form: geographic_unit, npi_name, start_date
           data frame confirmed_cases containing rows of the form: geographic_unit, cases, date
           
    Output: matrix where rows represent geographic units, columns represent npi's, 
               values represent the length of adoption period (in days or weeks) 
               of an npi at a geographic unit G, up to D days after the Nth confirmed case in G.
    '''

    colnames = npi_data_df.columns
    assert len(colnames) == 3
    location_col, npi_col, start_date_col = colnames

    ## extract the start date of each npi in each country or county
    #npi_adoption_date_matrix = npi_data_df.pivot(index='county_state', columns='npi', values='start_date')
    npi_adoption_date_matrix = npi_data_df.pivot(index=location_col, columns=npi_col, values=start_date_col)

    assert set(reference_date.index) - set(npi_adoption_date_matrix.index) == set()

    # calculate number of days that each measure was adopted prior to the reference date
    num_days_since_ref_date = npi_adoption_date_matrix.apply(lambda x: reference_date-x, axis=0)
    npi_aligned_matrix = num_days_since_ref_date.applymap(lambda x: x.days if (not math.isnan(x.days) and x.days>0) else 0)

    return npi_aligned_matrix
