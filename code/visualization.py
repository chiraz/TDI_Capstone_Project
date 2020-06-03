import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

import feature_engineering


# ========================================================================================== #
# DATA VISUALIZATION FUNCTIONS
# ========================================================================================== #


# Foo1, for lack of a better name!
def foo1(npi_data_df, confirmed_cases_df, N, K):
    '''
    N: use Nth case as reference date
    K: number of selected counties for display
    '''

    # Calculate date of Nth case for each county
    reference_date = feature_engineering.date_Nth_case(confirmed_cases_df, N=N)

    # start date of NPI adoption as number of days since Nth case
    npi_data_df['start_day'] = npi_data_df.apply(lambda row: (row.start_date - reference_date[row.county_state]).days, axis=1)

    cum_num_npis_matrix = npi_data_df.pivot_table(index='county_state', columns='start_day', values='npi', aggfunc='count').fillna(value=0).cumsum(axis=1)
    cum_num_npis_matrix.columns.name = 'start_day'

    confirmed_cases_ts = feature_engineering.get_confirmed_cases_time_series_(confirmed_cases_df)
    top_counties = confirmed_cases_ts.max(axis=0).sort_values(ascending=False).head(K).index

    df_temp = cum_num_npis_matrix.loc[top_counties]
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_temp, fmt="d", linewidths=.1, square=True,
                linecolor='white', annot=False, cmap="viridis", cbar=True)
    plt.ylabel('County')
    plt.xlabel(f'Days since {N}th case')
    plt.title(f'Cumulative number of adopted NPIs in top {K} counties', y=1.05, size=10)
    plt.show()


def foo2(npi_data_df, confirmed_cases_df, N):

    # Calculate date of Nth case for each county
    reference_date = feature_engineering.date_Nth_case(confirmed_cases_df, N=N)
    
    # Number of adopted NPI's by this date
    npi_data_df['npi_started_before_Nth_case'] = npi_data_df.apply(lambda x: x.start_date<=reference_date[x.county_state], axis=1)
    num_npis_started_before_Nth_case = npi_data_df.pivot(index='county_state', columns='npi', values='npi_started_before_Nth_case').fillna(value=False).sum(axis=1)

    # Alternative method -- JUST FOR DEBUGGING
    #npi_data_df['start_day'] = npi_data_df.apply(lambda row: (row.start_date - reference_date[row.county_state]).days, axis=1)
    #cum_num_npis_matrix = npi_data_df.pivot_table(index='county_state', columns='start_day', values='npi', aggfunc='count').fillna(value=0).cumsum(axis=1)
    #col = cum_num_npis_matrix.columns[cum_num_npis_matrix.columns==0]
    #num_npis_started_before_Nth_case_ = cum_num_npis_matrix[col].iloc[:,0]
    #assert all(num_npis_started_before_Nth_case_ == num_npis_started_before_Nth_case)

    confirmed_cases_ts = feature_engineering.get_confirmed_cases_time_series_(confirmed_cases_df)

    u = pd.concat([num_npis_started_before_Nth_case, confirmed_cases_ts.max(axis=0)], axis=1)
    u.columns = ['cum_num_adopted_npis', 'total_num_cases']

    sns.boxplot(x='cum_num_adopted_npis', y='total_num_cases', data=u)
    plt.title(f'Prior to {N}th case')
    plt.ylabel('total cases')
    plt.xlabel('')
    #plt.xlabel('num adopted NPI\'s')
    plt.yscale('log')


def foo3(npi_data_df, confirmed_cases_df, N):

    # Calculate date of Nth case for each county
    reference_date = feature_engineering.date_Nth_case(confirmed_cases_df, N=N)
    
    # Number of adopted NPI's by this date
    npi_data_df['npi_started_before_Nth_case'] = npi_data_df.apply(lambda x: x.start_date<=reference_date[x.county_state], axis=1)
    num_npis_started_before_Nth_case = npi_data_df.pivot(index='county_state', columns='npi', values='npi_started_before_Nth_case').fillna(value=False).sum(axis=1)

    new_confirmed_cases_ts = feature_engineering.get_new_confirmed_cases_(confirmed_cases_df, smoothing_ws=3)

    u = pd.concat([num_npis_started_before_Nth_case, new_confirmed_cases_ts.max(axis=0)], axis=1)
    u.columns = ['cum_num_adopted_npis', 'peak_num_new_cases']

    sns.boxplot(x='cum_num_adopted_npis', y='peak_num_new_cases', data=u)
    plt.title(f'Prior to {N}th case')
    plt.ylabel('peak new cases')
    plt.xlabel('')
    #plt.xlabel('num adopted NPI\'s')
    plt.yscale('log')


def foo4(npi_data_df, confirmed_cases_df, N, D, num_periods=5, smoothing_ws=3):

    ## calculate date of Nth case for each county
    reference_date = feature_engineering.date_Nth_case(confirmed_cases_df, N=N)

    # Number of adopted NPI's by reference date
    npi_data_df['npi_started_before_Nth_case'] = npi_data_df.apply(lambda x: x.start_date<=reference_date[x.county_state], axis=1)
    num_npis_started_before_Nth_case = npi_data_df.pivot(index='county_state', columns='npi', values='npi_started_before_Nth_case').fillna(value=False).sum(axis=1)

    ## Calculate infection rate time series for each county
    infection_rate_ts = feature_engineering.calc_infection_rate_time_series_(confirmed_cases_df, num_periods=num_periods, smoothing_ws=smoothing_ws)

    ## infection rate D days after Nth case
    target_infection_rate_ts = feature_engineering.align_time_series_(infection_rate_ts, reference_date).iloc[D,:].T
    
    ## trimming
    #q = target_infection_rate_ts.quantile(q=.9)
    #target_infection_rate_ts = target_infection_rate_ts.map(lambda x: x if x<q else q)

    u = pd.concat([num_npis_started_before_Nth_case, target_infection_rate_ts], axis=1)
    u.columns = ['cum_num_adopted_npis', 'infection_rate']

    sns.boxplot(x='cum_num_adopted_npis', y='infection_rate', data=u)
    #plt.title(f'IR@{D} days after {N}th case vs. num adopted NPI prior to {N}th case')
    #plt.ylabel('trimmed infection rate')
    #plt.xlabel('num adopted NPI\'s')
    plt.ylabel('')
    plt.xlabel('')


def visualize_NPI_adoption():
    #  Adopted measures by location and date

    pass



def visualize_confirmed_cases():
    #  Adopted measures by location and date

    pass


def visualize_infection_rates():
    pass


