import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

import math
import numpy as np
import pandas as pd
from scipy.stats import kruskal


from feature_engineering import FeatureEngineering


class DataViz(FeatureEngineering):
    def __init__(self, *args, location_type="Location", location_type_plural="Locations"):
        #super(DataViz, self).__init__(*args)
        FeatureEngineering.__init__(self, *args)
        self.location_type = location_type
        self.location_type_plural = location_type_plural


    def nth_case_dates(self):
        ''' Calculate and compare dates of 10th and 100th cases for each location'''

        reference_date1 = self.calc_date_Nth_case_(N=10)
        reference_date2 = self.calc_date_Nth_case_(N=100)
        reference_date1.value_counts().sort_index().cumsum().plot(rot=20, label='N=10')
        reference_date2.value_counts().sort_index().cumsum().plot(rot=20, label='N=100')
        plt.title(f'Cumulative number of {self.location_type_plural.lower()} that reached Nth case, N=10, 100')
        plt.xlabel('Date')
        plt.legend(loc='best')
        plt.show()

        (reference_date2-reference_date1).map(lambda x: x.days).plot.hist(density=False, cumulative=False)
        plt.ylabel(f'Number {self.location_type_plural}')
        plt.xlabel('Num Days')
        plt.title(f'Number of days between 10th and 100th cases in a {self.location_type.lower()}')
        plt.show()


    def num_daily_new_cases_over_time(self, smoothing_ws=3, K=50, n=0):
        ''' Visualize number of daily new cases over time as a heatmap.
        Inputs: 
          K: number of locations (having highest number of confirmed cases) that will be displayed.
        '''

        ## Get time series
        confirmed_cases_ts = self.get_confirmed_cases_time_series()
        new_confirmed_cases_ts = self.get_new_cases_time_series(smoothing_ws=smoothing_ws)

        ## Select only locations with the highest number of cases
        select_locations = confirmed_cases_ts.max(axis=0).sort_values(ascending=False).head(K).index
        new_confirmed_cases_ts = new_confirmed_cases_ts[select_locations]

        ## Normalize each time series to the range [0,1]
        new_confirmed_cases_ts = new_confirmed_cases_ts.apply(lambda x: x/max(x))

        ## Display as a heatmap with locations as rows
        df_temp = new_confirmed_cases_ts.iloc[n:,:].T
        df_temp.columns = df_temp.columns.strftime("%m-%d")
        plt.figure(figsize=(15, 15))
        sns.heatmap(df_temp, fmt="d", linewidths=.5, square=True,
                    linecolor='white', annot=False, cmap="viridis", cbar=True, cbar_kws={"orientation": "horizontal"})
        plt.ylabel(f"{self.location_type_plural}, in decreasing order of total cases", size=12)
        plt.xlabel('Date', y=1.1, size=12)
        plt.title(f'Normalized number of daily new cases in {K} worst-affected {self.location_type_plural.lower()}', y=1.1, size=17)
        plt.show()


    def infection_rate_over_time(self, num_periods, smoothing_ws=3, q=.95, K=50, n=0):

        ## Get time series
        confirmed_cases_ts = self.get_confirmed_cases_time_series()
        infection_rate_ts = self.get_infection_rate_time_series(num_periods=num_periods, smoothing_ws=smoothing_ws, q=q)

        ## Select only locations with the highest number of cases
        select_locations = confirmed_cases_ts.max(axis=0).sort_values(ascending=False).head(K).index
        infection_rate_ts = infection_rate_ts[select_locations]

        ## Display as a heatmap with locations as rows
        df_temp = infection_rate_ts.iloc[n:,:].T
        df_temp.columns = df_temp.columns.strftime("%m-%d")
        plt.figure(figsize=(15, 15))
        sns.heatmap(df_temp, fmt="d", linewidths=.5, square=True,
                    linecolor='white', annot=False, cmap="viridis", cbar=True, cbar_kws={"orientation": "horizontal"})
        plt.ylabel(f"{self.location_type_plural}, in decreasing order of total cases", size=12)
        plt.xlabel('Date', size=12)
        plt.title(f'{num_periods}-day Infection Rate in {K} worst-affected {self.location_type_plural.lower()}', y=1.1, size=17)
        plt.show()


    def num_adopted_NPIs_distribution(self):
        num_adopted_npis_per_county = self.calc_npi_start_date_matrix_().notnull().sum(axis=1)

        plt.subplot(2,1,1)
        num_adopted_npis_per_county.plot.hist()
        #plt.xlabel('Number adopted NPI\'s')
        plt.ylabel(f'Number {self.location_type_plural.lower()}')
        plt.title(f'Distribution of number NPI\'s eventually adopted in a {self.location_type.lower()}')
        #plt.show()

        plt.subplot(2,1,2)
        num_adopted_npis_per_county.plot.hist(density=True, cumulative=True)
        plt.xlabel('Number adopted NPI\'s')
        plt.ylabel(f'Number {self.location_type_plural.lower()}')
        plt.title(f'Cumulative distribution')  #  of number of adopted NPI\'s at a single {self.location_type.lower()}

        plt.tight_layout()
        plt.show()        


    def percent_NPI_adoption(self, K=7):
        '''Visualize percent adoption for each NPI by location and over time
            K : number of top adopted NPI's that will be displayed
        '''

        start_date_matrix = self.calc_npi_start_date_matrix_()
        npi_percent_adoption_df = start_date_matrix.notnull().mean()*100
        num_locations = start_date_matrix.index.nunique()

        npi_percent_adoption_df.sort_values(ascending=False).plot.barh()
        plt.title(f'% {self.location_type_plural} that eventually adoped each NPI (n = {num_locations})')
        plt.show()

        top_adopted_npis = npi_percent_adoption_df.sort_values(ascending=False).head(K).index

        for x in top_adopted_npis:
            npi_data_col = self.get_npi_col()
            npi_start_date_col = self.get_npi_start_date_col()
            npi_location_col = self.get_npi_location_col()

            idx = npi_data_col == x
            npi_start_date_col.loc[idx].value_counts(normalize=False)\
                                       .map(lambda x: x/npi_location_col.nunique())\
                                       .sort_index().cumsum().plot(rot=30, label=x)

            plt.ylabel(f'% {self.location_type_plural.lower()}')
            plt.xlabel('Date')
            plt.title(f'Cumulative % {self.location_type_plural.lower()} that adopted NPI vs. time (top {K} NPIs only)', size=15)

        plt.legend(loc='best')
        plt.show()


    def num_adopted_NPIs_over_time(self, K=50):
        ''' Visualize number of adopted_NPIs per location over time as a heatmap.
        Inputs: K: number of selected counties for display
        '''

        num_adopted_npis_matrix = self.get_num_npis_matrix()

        ## Select locations to display based number of confirmed cases
        confirmed_cases_ts = self.get_confirmed_cases_time_series()
        select_locations = confirmed_cases_ts.max(axis=0).sort_values(ascending=False).head(K).index

        ## Display heatmap
        df_temp = num_adopted_npis_matrix.loc[select_locations]
        df_temp.columns = df_temp.columns.strftime("%m-%d")
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_temp, fmt="d", linewidths=.1, square=True,
                    linecolor='white', annot=False, cmap="viridis", cbar=True)  # , cbar_kws={"orientation": "horizontal"}
        plt.ylabel(f'{self.location_type}')
        plt.xlabel('Date')
        plt.title(f'Cumulative num adopted NPIs in {K} worst-affected {self.location_type_plural.lower()}', y=1.05, size=15)
        plt.show()

    def num_adopted_NPIs_aligned_by_Nth_case(self, N, K):
        '''
        Inputs:
        N: use date of Nth case as reference date (date 0) for each location
        K: number of selected counties for display
        '''

        normalized_num_adopted_npis_matrix = self.get_normalized_num_npis_matrix(N)

        ## Select locations to display based number of confirmed cases
        confirmed_cases_ts = self.get_confirmed_cases_time_series()
        select_locations = confirmed_cases_ts.max(axis=0).sort_values(ascending=False).head(K).index

        ## Display heatmap
        df_temp = normalized_num_adopted_npis_matrix.loc[select_locations]
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_temp, fmt="d", linewidths=.1, square=True,
                    linecolor='white', annot=False, cmap="viridis", cbar=True)
        plt.ylabel('County')
        plt.xlabel(f'Days since {N}th case')
        plt.title(f'Cumulative num adopted NPIs in {K} worst-affected {self.location_type_plural.lower()}', y=1.05, size=12)
        plt.show()



    def total_cases_vs_num_adopted_npis(self, N, K):

        # Number of adopted NPI's by reference date
        normalized_num_adopted_npis_matrix = self.get_normalized_num_npis_matrix(N)
        num_npis_started_before_Nth_case = normalized_num_adopted_npis_matrix.loc[:,0]
        assert all(num_npis_started_before_Nth_case.notnull())

        ## get the time series of number of confirmed cases for each location
        confirmed_cases_ts = self.get_confirmed_cases_time_series()

        ## get the time series of number of confirmed cases K days after Nth case
        aligned_ts = self.align_time_series(confirmed_cases_ts, N)
        target_confirmed_cases_ts = aligned_ts.iloc[K,:].T

        assert num_npis_started_before_Nth_case.shape[0] >= target_confirmed_cases_ts.count()

        if any(target_confirmed_cases_ts.isnull()):
            idx1 = target_confirmed_cases_ts.notnull()
            target_confirmed_cases_ts = target_confirmed_cases_ts.loc[idx1]
            num_npis_started_before_Nth_case = num_npis_started_before_Nth_case.loc[target_confirmed_cases_ts.index]

        assert num_npis_started_before_Nth_case.shape == target_confirmed_cases_ts.shape
        assert all(num_npis_started_before_Nth_case.index == target_confirmed_cases_ts.index)

        ## Visualize

        temp_df = pd.concat([num_npis_started_before_Nth_case, target_confirmed_cases_ts], axis=1)
        temp_df.columns = ['num_adopted_npis', 'total_num_cases']

        sns.boxplot(x='num_adopted_npis', y='total_num_cases', data=temp_df)
        plt.title(f'N={N}, K={K}', size=15)
        plt.ylabel('total num cases', fontsize=15)
        plt.xlabel('total num adopted NPI\'s', fontsize=15)
        plt.yscale('log')


    def total_cases_vs_npi_num_days(self, N, K, show_plots=True):
        # number of adopted days for each npi in each county or country
        npi_num_days_matrix = self.get_NPI_agg_features2(N, K)
        
        ## get the time series of number of confirmed cases for each location
        confirmed_cases_ts = self.get_confirmed_cases_time_series()

        ## get the time series of number of confirmed cases K days after Nth case
        aligned_ts = self.align_time_series(confirmed_cases_ts, N)
        target_confirmed_cases_ts = aligned_ts.iloc[K,:].T

        assert npi_num_days_matrix.shape[0] >= target_confirmed_cases_ts.count()

        if any(target_confirmed_cases_ts.isnull()):
            idx1 = target_confirmed_cases_ts.notnull()
            target_confirmed_cases_ts = target_confirmed_cases_ts.loc[idx1]
            npi_num_days_matrix = npi_num_days_matrix.loc[target_confirmed_cases_ts.index]

        assert npi_num_days_matrix.shape[0] == target_confirmed_cases_ts.shape[0]
        assert all(npi_num_days_matrix.index == target_confirmed_cases_ts.index)

        
        ## Visualize
        ## TO DO: add mutual information score in the title

        if show_plots:
            plt.figure(figsize=(20, 20))
            n = int(npi_num_days_matrix.shape[1] // 3)
            i = 1

            for npi_name, npi_col in npi_num_days_matrix.iteritems():
                plt.subplot(3,n,i)
                #sns.boxplot(x=npi_col, y=target_confirmed_cases_ts)
                plt.scatter(x=npi_col, y=target_confirmed_cases_ts, c='b')
                plt.yscale('log')
                plt.xlabel('num days', fontsize=15)
                plt.ylabel('num cases', fontsize=15)
                plt.title(npi_name, fontsize=15)
                i = i+1

            plt.tight_layout()
            plt.show()

            return None

        else:
            return npi_num_days_matrix, target_confirmed_cases_ts
    

# ---------------------------- UNUSED METHODS --------------------------------------------- #

    def __peak_new_cases_vs_num_adopted_npis__(self, N, smoothing_ws=3):

        normalized_num_adopted_npis_matrix = self.get_normalized_num_npis_matrix(N)
        num_npis_started_before_Nth_case = normalized_num_adopted_npis_matrix.loc[:,0]
  
        new_confirmed_cases_ts = self.get_new_cases_time_series(smoothing_ws=smoothing_ws)

        u = pd.concat([num_npis_started_before_Nth_case, new_confirmed_cases_ts.max(axis=0)], axis=1)
        u.columns = ['cum_num_adopted_npis', 'peak_num_new_cases']

        sns.boxplot(x='cum_num_adopted_npis', y='peak_num_new_cases', data=u)
        plt.title(f'Prior to {N}th case')
        plt.ylabel('peak daily new cases')
        plt.xlabel('num adopted NPI\'s')
        plt.yscale('log')
        
        

    def __infection_rate_vs_num_adopted_npis__(self, N, D, num_periods=5, smoothing_ws=3, q=.95):

        # Number of adopted NPI's by reference date
        normalized_num_adopted_npis_matrix = self.get_normalized_num_npis_matrix(N)
        num_npis_started_before_Nth_case = normalized_num_adopted_npis_matrix.loc[:,0]

        ## Calculate infection rate time series for each county
        infection_rate_ts = self.get_infection_rate_time_series(num_periods=num_periods, smoothing_ws=smoothing_ws, q=0)

        ## infection rate D days after Nth case
        target_infection_rate = self.align_time_series(infection_rate_ts, N).iloc[D,:].T

        ## trimming
        qv = np.nanquantile(target_infection_rate.values, q=q)
        target_infection_rate_trimmed = target_infection_rate.map(lambda x: x if x<qv or math.isnan(x) else qv)

        ## Visualize

        temp_df = pd.concat([num_npis_started_before_Nth_case, target_infection_rate_trimmed], axis=1)
        temp_df.columns = ['cum_num_adopted_npis', 'infection_rate']

        sns.boxplot(x='cum_num_adopted_npis', y='infection_rate', data=temp_df)
        plt.title(f'Infection Rate {D} days after {N}th case vs. num adopted NPI\'s prior to {N}th case', size=12)
        plt.ylabel('infection rate')
        plt.xlabel('num adopted NPI\'s')


        ## Non-parametric hypothesis test of independence

        # divide infection rate into three groups: 0, 1-2, 3-9
        data1 = target_infection_rate.loc[num_npis_started_before_Nth_case==0]
        data2 = target_infection_rate.loc[(num_npis_started_before_Nth_case==1) | (num_npis_started_before_Nth_case==2)]
        data3 = target_infection_rate.loc[num_npis_started_before_Nth_case>=3]

        print("\nHypothesis Testing:")
        print(f"\tNum elements in each group:{(data1.shape[0],data2.shape[0],data3.shape[0])}")

        stat, p = kruskal(data1.values, data2.values, data3.values)
        print('\tStatistics=%.5f, p=%.5f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('\tSame distributions (fail to reject H0)')
        else:
            print('\tDifferent distributions (reject H0)')        
