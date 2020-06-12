# TDI_Capstone_Project

This repository contains the final submission of my Data Incubator (TDI) Capstone Project.

# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #

## Checklist of TDI requirements

### A clear business objective

The goal of this project is to assess the effectiveness of governmental Non-Pharmaceutical Interventions 
(henceforth referred to as **NPIs**) in response to the recent Covid-19 pandemic. 
NPIs include measures like the closing of schools, travel bans, and restriction on public gatherings. 
In the early stages of an epidemic, government officials will scramble to come up with a 
swift response strategy that strikes the right balance between mitigating the public health risk of the disease 
and minimizing socio-economic consequences of strict intervention measures. Their decisions are typically based on 
recommendations of public health advisors, which are in turn informed by sophisticated epidemiological and behavioral models. 

In this project, we propose to use a data-driven machine learning approach based on actual data of NPI 
strategies of different governments in order to:

1. determine the impact of different NPI's, as well as of their timing, on the progress of Covid19.
2. make informed recommendations of which and/or how many NPI's are best to use and *when* they should be rolled out.


### Data ingestion

Our analysis is based on publicly available data about NPI adoptions of various governments throughout the world. 
Specifically, we use two different, independently collected NPI datasets:

- A dataset of 150+ countries and 13 NPIs, collected by researchers at the University of Oxford and can be freely downloaded from:
https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv


- A dataset of 330 US counties and 9 NPIs, collected by a New York-based consulting firm (called Keystone Strategy) in 
collaboration with Stanford University, and is available for download at:
https://www.keystonestrategy.com/coronavirus-covid19-intervention-dataset-model/



### Visualizations

The first major part of our analysis (Section entitled "Descriptive Analysis" in the notebook) consists 
of a detailed exploratory analysis of the datasets using various forms of visualizations 
such as scatterplots and heatmaps, in order to:

1. visualize the distributions of the quantities of interest, namely the progression of the disease and NPI adoption variables.
2. visualize the relationship between these variables.
3. inspire appropriate data preprocessing and feature engineering methods for predictive analysis.

### Demonstration of at least one of the following: a. Machine learning b. Distributed computing c. An interactive website

The second major part of our analysis (Section entitled "Predictive Analysis" in the notebook) consists of 
building a machine learning model of the relationship between the progression of Covid19 disease and NPI adoption in a county. 
The main use case of this model is not forecasting the progress of the disease (there are better ways of doing that), 
but rather to be able to conduct what-if scenarios, that is to determine the outcome of alternative intervention strategies, 
and hence to be able to plan better strategies in the future. This is called *prescriptive analysis*.

**Target and feature variables**

The target and features of our model are:

- **Target**: *log* of total number of confirmed cases at $K$ days after the $N$th confirmed case in a county.

- **Features**: number of days from adoption of each NPI by a county till the $N$th confirmed case in that county, one variable per NPI.

where $N$ and $K$ are fixed integer parameters to be chosen wisely. These representations are explained in more detail in the Feature Engineering Section below.


**Regression models**

Since the target variable is continuous, we have experimented with a number of regression modeling techniques, and eventually selected the best one based on cross-validation performance:


- Linear ridge regression
- Support vector machines with linear kernel
- Support vector machines with RBF kernel
- K-Nearest Neighbors
- Random forests
- Stacked ensemble model of the above models

**Performance metric**

Because the target variable contains some outliers, we have opted to use the **mean absolute error** (MAE) as the primary performance metric
in model selection.


### Deliverable

Our main deliverable is a Jupyter notebook that contains a cleaned up version of all the data analyses 
(both descriptive and predictive) we've performed to meet the business objective of the project.

# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #

## Glossary and definitions

- NPI: Non-pharmaceutical intervention.
- Total number of confirmed cases, denoted $C(t)$: cumulative number of confirmed cases up to day $t$. 
This quantity depends on the number of tests carried out and is generally much smaller than the *true* number of cases, 
but is considered to be a pretty good surrogate.
- Number of daily *new* cases, calculated as $C(t) - C(t-1)$, since the number of cases is observed daily.
- Infection rate: percent change in number of confirmed cases, calculated as $\frac{C(t) - C(t-p)}{C(t-1)}$, 
where $p$ is a small integer, usually larger than 1 to smooth noise.

# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #

## Machine learning pipeline

### Data preprocessing

Our two datasets differ quite a bit in terms of the number and coding of covered NPI's. For example the first dataset provides 
information about the strictness level of each NPI, while the second one doesn't. In order to build a uniform framework for 
analyzing both datasets, in the preprocessing stage we extract the following 3 basic pieces of data from each dataset :

1. location of the NPI (country or US county).
2. name and description of each NPI.
3. adoption date of each NPI.


### Feature extraction and engineering

The underlying goal of this project is to determine whether there is a significant relationship or association between the state of disease progression in a country or county, and the timing of NPI adoptions in that country, and subsequently to quantity this relationship. Simply put, countries that have acted sufficiently early (relative to the onset of the disease in that county), were they on average affected by the disease significantly less than those that haven't?  

To answer this question via machine learning, we start by finding an appropriate representation of the two properties of interest. Our representations are normalized for variation in the onset of the disease in different countries or counties. Specifically, let $t_0(c)$ denote the date of the $N$th case in a country or county $c$, where $N$ is a fixed parameter.


#### Representation of disease progression

We have considered the two alternative representations: 

1. *log* of total number of confirmed cases at $K$ days after $t_0(c)$ date. 
2. infection rate at $K$ days after $t_0(c)$ date.

where $K$ is a fixed positive integer that accounts for the lag between when NPI's are adopted 
and when their effect might be observed on the progression of the disease. 
The log transformation in the first representation helps make the 
distribution of the variable more symmetric, which is very important in predictive analysis.

Eventually we have opted for the first representation because it gave better results in the 
predictive analysis. This is not surprising because infection rate is more 
sensitive to inaccuracies in number of daily cases. 


#### Representation of NPI adoption

We seek a simple representation for each county or country $c$ that captures *whether* and *when* each NPI has been adoptd by that country or county. 
Our representation of choice consists of $n$ features, one feature per NPI. 
The $i$th feature value is the number of days from the adoption of the $i$th NPI till date $t_0(c)$. 
These feature values are always non-negative, and are 0 if an NPI was either not adopted at all by $c$ or adopted after the reference date $t_0(c)$. 

This representation turned out to have best predictive power of disease progression compared to other representations that we experimented with.


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #

## Key results


