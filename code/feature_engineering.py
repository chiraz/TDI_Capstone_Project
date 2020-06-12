import numpy as np
import pandas as pd
import datetime
import math

# TO DO: REPLACE ALL ASSERTIONS WITH THROW EXCEPTIONS
# TO DO: RENAME AS STATELESS_DATA_MUNGING_OR_WRANGLING

class FeatureEngineering():
    def __init__(self, npi_data_df, confirmed_cases_df):
        '''
        Inputs: 2 data frames containing preprocessed data
          npi_data_df: data frame containing rows of the form: geographic location, npi_name, start_date
          confirmed_cases_df : data frame containing rows of the form: geographic location, date, cumulative number of cases
        '''

        ## preprocessed data
        self.npi_data_df = npi_data_df
        self.confirmed_cases_df = confirmed_cases_df

        colnames = self.npi_data_df.columns
        assert len(colnames) == 3
        self.npi_location_col, self.npi_col, self.npi_start_date_col = colnames

        colnames = self.confirmed_cases_df.columns
        assert len(colnames) == 3
        self.confirmed_cases_location_col, self.confirmed_cases_date_col, self.confirmed_cases_num_col = colnames


        ## initialize transformed data
        self.confirmed_cases_ts = None   # time series of confirmed cases at each location
        #self.new_cases_ts = None    # time series of daily new confirmed cases at each location
        #self.infection_rate_ts = None   # time series of infection rate at each location

    def get_npi_location_col(self):
        return self.npi_data_df[self.npi_location_col]

    def get_npi_col(self):
        return self.npi_data_df[self.npi_col]

    def get_npi_start_date_col(self):
        return self.npi_data_df[self.npi_start_date_col]

    def get_confirmed_cases_time_series(self):
        if self.confirmed_cases_ts is None:
            self.calc_confirmed_cases_ts_()
        return self.confirmed_cases_ts

    def get_new_cases_time_series(self, smoothing_ws=3):
        '''smoothing_ws : window size for smoothing infection rate time series.'''
        return self.calc_new_cases_ts_(smoothing_ws=smoothing_ws)

    def align_time_series(self, ts, N):
        '''
        Input: 
            ts : matrix of locations and time series; each row is a time series of a location.
            N : align each row by the date of Nth case.
        Output: two matrix: where rows represent geographic location, columns represent number of 
                    days since the Nth confirmed case in a geographic location, and values
                    represent the number of confirmed cases, and the infection rate, respectively.
        '''

        reference_date = self.calc_date_Nth_case_(N)

        extract_ts = lambda x,date:  x.loc[x.index>=date].reset_index(drop=True)

        d = dict()
        for colname,coldata in ts.iteritems():
            d[colname] = extract_ts(coldata,reference_date[colname])
        aligned_ts = pd.DataFrame(d)
        aligned_ts.index.name = 'num_days'
        aligned_ts.columns.name = ts.columns.name

        return aligned_ts

    def get_num_npis_matrix(self):
        '''Returns a matrix, one row per location, one column per date.
        where (i,j)th value is the number of NPI's adopted so far at
        the ith location as of date j.
        '''

        num_npis_matrix = self.npi_data_df.pivot_table(index=self.npi_location_col, 
                                                       columns=self.npi_start_date_col, 
                                                       values=self.npi_col, aggfunc='count')\
                                         .fillna(value=0)\
                                         .cumsum(axis=1)
        num_npis_matrix.columns.name = 'adoption_date'
        return num_npis_matrix


    def get_normalized_num_npis_matrix(self, N):
        '''Returns a matrix, rows are locations, 
        columns are number of days relative to date of Nth case,
        and the (i,j)th value is the number of NPI's adopted so far at
        the ith location j days after it reached its Nth confirmed case,
        where N is an input parameter.
        The date of the Nth confirmed case serves as a reference date
        for normalizing variations in start dates of the disease.
        '''

        # Calculate date of Nth case for each county
        reference_date = self.calc_date_Nth_case_(N=N)

        # start date of NPI adoption as number of days since Nth case
        npi_data_df = self.npi_data_df.copy()
        npi_data_df['start_day'] = self.npi_data_df.apply(lambda row: (row.start_date - reference_date[row.county_state]).days, axis=1)

        normalized_num_npis_matrix = npi_data_df.pivot_table(index=self.npi_location_col, 
                                                             columns='start_day', 
                                                             values=self.npi_col, aggfunc='count')\
                                                 .fillna(value=0)\
                                                 .cumsum(axis=1)

        normalized_num_npis_matrix.columns.name = 'adoption_day'
        return normalized_num_npis_matrix


    def get_target_vector2(self, N, D):
        ''' Number of confirmed cases D days after Nth confirmed case at every location. '''
        
        ## Number of confirmed cases time series
        confirmed_cases_ts = self.get_confirmed_cases_time_series()

        ## Number of confirmed cases D days after Nth case
        target_confirmed_cases_ts = self.align_time_series(confirmed_cases_ts, N).iloc[D,:].T

        return np.log10(target_confirmed_cases_ts)


    def get_NPI_agg_features2(self, N, D=0):
        ''' 
          Inputs:
            N: number of cases parameter.
            D: number of days parameter.
          Returns : a matrix, where rows represent geographic locations, columns represent 
            NPIs, and the (i,j)th value is the number of days that the jth NPI was in effect
            up to D days after Nth confirmed case. Thus these features implicity encode for  
            each (location,NPI) pair both whether and when the NPI was adopted at that location.
        '''

        reference_date0 = self.calc_date_Nth_case_(N)
        reference_date1 = reference_date0 + datetime.timedelta(days=D)   # date of Nth case + D days
        
        ## extract the start date of each npi in each geographic location
        npi_adoption_date_matrix = self.calc_npi_start_date_matrix_()

        assert set(reference_date1.index) - set(npi_adoption_date_matrix.index) == set()

        # calculate number of days that each measure was adopted prior to the reference date
        num_days_since_ref_date = npi_adoption_date_matrix.apply(lambda x: reference_date1-x, axis=0)
        npi_feature_matrix = num_days_since_ref_date.applymap(lambda x: x.days if (not math.isnan(x.days) and x.days>0) else 0)
        assert npi_feature_matrix.isnull().sum().sum() == 0
        npi_feature_matrix.loc[reference_date0.isnull(),:] = math.nan

        ## FOR DEBUGGING
        #print(reference_date0.isnull().sum(), reference_date1.isnull().sum())
        #print(npi_feature_matrix.isnull().sum(axis=1).sort_values(ascending=False).head(10))


        return npi_feature_matrix


    #===========================================================================#


    def calc_npi_start_date_matrix_(self):
        #npi_data_df.pivot(index='county_state', columns='npi', values='start_date')
        return self.npi_data_df.pivot(index=self.npi_location_col,
                                      columns=self.npi_col,
                                      values=self.npi_start_date_col)

    def calc_date_Nth_case_(self, N):
        ''' returns the date of Nth case for each geographic location'''
        ''' fixed bug on 6/9/2020 '''

        if self.confirmed_cases_ts is None:
            self.calc_confirmed_cases_ts_()

        def helper_func_(x, N):
            '''Input: x is a time series. N is a value.
               Returns the date such that x[date] ~= N 
            '''

            ##OLD VERSION
            ##return x[x>=N].index.min()

            ## MORE CORRECT VERSION, as of 6/9/2020
            
            if N > x.max():
                return math.nan

            #elif N < x.min():
            #    return x[x>0].index.min()

            elif (x == N).sum():
                return x[x == N].index.min()

            else:
                idx1 = x<N
                idx2 = x>N
                assert idx1.sum() and idx2.sum()

                i1 = x[idx1].index.max() 
                i2 = x[idx2].index.min()
                x1 = x[i1]
                x2 = x[i2]
                delta = int(round((N-x1)/(x2-x1)*(i2-i1).days))
                return i1 + pd.Timedelta(delta, unit='days')

        return self.confirmed_cases_ts.apply(lambda x: helper_func_(x,N), axis=0)


    def calc_confirmed_cases_ts_(self):
        '''
        re-calculates and sets the value of self.confirmed_cases_ts
        '''

        ## extract non-aligned confirmed cases matrix; the columns in this matrix correspond to real start dates.
        confirmed_cases_ts = self.confirmed_cases_df.pivot(index=self.confirmed_cases_date_col, 
                                                           columns=self.confirmed_cases_location_col, 
                                                           values=self.confirmed_cases_num_col)

        ## fill-in leading missing values with 0's in each time series
        # because they indicate 0 confirmed cases at that date.

        confirmed_cases_ts.fillna(value=0, inplace=True)
        assert confirmed_cases_ts.isnull().sum().sum() == 0
        
        self.confirmed_cases_ts = confirmed_cases_ts


    def calc_new_cases_ts_(self, smoothing_ws):
        if self.confirmed_cases_ts is None:
            self.calc_confirmed_cases_ts_()

        new_cases_ts = self.confirmed_cases_ts.diff(periods=1, axis=0).rolling(window=smoothing_ws).mean()

        # fill-in leading missing values; they are just an artifact of differencing.
        new_cases_ts.fillna(method='backfill', inplace=True)
        assert new_cases_ts.isnull().sum().sum() == 0
        
        return new_cases_ts
        
# ---------------------------- UNUSED METHODS --------------------------------------------- #


    def __get_target_vector1__(self):
        ''' Total number of confirmed cases at every location. '''
        confirmed_cases_ts = self.get_confirmed_cases_time_series()
        return np.log10(confirmed_cases_ts.max(axis=0))

    def __get_NPI_agg_features1__(self, N, D):
        '''
        Inputs:
          N: number of cases that characterize onset of disease at a location.
          D: number of days after date of Nth case.
        Returns : a vector, one value per location, where ith value is the number 
          of NPI's adopted at location i up to D days after date of Nth confirmed case.
        '''

        normalized_num_npis_matrix = self.get_normalized_num_npis_matrix(N)
        return normalized_num_npis_matrix.loc[:,D]


    def __get_infection_rate_time_series__(self, num_periods, smoothing_ws=3, q=.95):
        '''
          num_periods : interval size in number of days for calculating infection rate :
               where InfectionRate(t) = (ConfirmedCases(t)-ConfirmedCases(t-num_periods)) / ConfirmedCases(t-num_periods)
          smoothing_ws : window size for smoothing infection rate time series.
        '''
        return self.__calc_infection_rate_ts__(num_periods=num_periods, smoothing_ws=smoothing_ws, q=q)


    def __calc_infection_rate_ts__(self, num_periods, smoothing_ws, q=0):

        if self.confirmed_cases_ts is None:
            self.calc_confirmed_cases_ts_()

        calc_change_rate = lambda x, n, w: x.fillna(method='ffill') \
                 .pct_change(periods=n, fill_method='ffill') \
                 .rolling(window=w).mean()  ##.median()

        infection_rate_ts = self.confirmed_cases_ts.apply(lambda x: calc_change_rate(x, n=num_periods, w=smoothing_ws), axis=0)

        ## trimming
        if q > 0:
            qv = np.nanquantile(infection_rate_ts.values, q=q)
            infection_rate_ts = infection_rate_ts.applymap(lambda x: x if x<qv or math.isnan(x) else qv)

        return infection_rate_ts
