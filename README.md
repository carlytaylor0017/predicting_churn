# Comparison of parametric and non-parametric models in predicting churn
## John Herr
## Joe Tustin
## Carly Wolfbrandt

### Table of Contents
1. [Objective](#objective)
2. [Exploratory Data Analysis](#eda)
    1. [Dataset](#dataset) 
    2. [Data Cleaning](#cleaning)
    3. [Feature Engineering](#engineering)
3. [Modelling](#model)
    1. [Model Pipeline](#pipeline)
    2. [Model Scoring](#scoring)
    3. [Random Forest](#rf)
    4. [Linear Regression](#lm)

## Objective <a name="objective"></a>

Use rideshare data set to help understand what factors are the best predictors for churn, and offer insights to help improve customer retention.

## Exploratory Data Analysis <a name="eda"></a>

### Dataset <a name="dataset"></a>

A ride-sharing company (Company X) is interested in predicting rider retention. To help explore this question, we used a sample dataset of a cohort of users who signed up for an account in January 2014. The data was pulled on July 1, 2014; we consider a user retained if they were “active” (i.e. took a trip) in the preceding 30 days (from the day the data was pulled). In other words, a user is "active" if they have taken a trip since June 1, 2014.

Here is a detailed description of the data:

***CATEGORICAL***
- `city`: city this user signed up in 
- `phone`: primary device for this user
- `luxury_car_user`: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise 

***NUMERICAL***
- `signup_date`: date of account registration; in the form `YYYYMMDD`
- `last_trip_date`: the last time this user completed a trip; in the form `YYYYMMDD`
- `avg_dist`: the average distance (in miles) per trip taken in the first 30 days after signup
- `avg_rating_by_driver`: the rider’s average rating by their drivers over all of their trips 
- `avg_rating_of_driver`: the rider’s average rating of their drivers over all of their trips 
- `surge_pct`: the percent of trips taken with surge multiplier > 1 
- `avg_surge`: The average surge multiplier over all of this user’s trips 
- `trips_in_first_30_days`: the number of trips this user took in the first 30 days after signing up 
- `weekday_pct`: the percent of the user’s trips occurring during a weekday

**Table 1**: Initial dataset 

|    |   avg_dist |   avg_rating_by_driver |   avg_rating_of_driver |   avg_surge | city           | last_trip_date      | phone   | signup_date         |   surge_pct |   trips_in_first_30_days | luxury_car_user   |   weekday_pct |
|---:|-----------:|-----------------------:|-----------------------:|------------:|:---------------|:--------------------|:--------|:--------------------|------------:|-------------------------:|:------------------|--------------:|
|  0 |       3.67 |                    5   |                    4.7 |        1.1  | King's Landing | 2014-06-17 00:00:00 | iPhone  | 2014-01-25 00:00:00 |        15.4 |                        4 | True              |          46.2 |
|  1 |       8.26 |                    5   |                    5   |        1    | Astapor        | 2014-05-05 00:00:00 | Android | 2014-01-29 00:00:00 |         0   |                        0 | False             |          50   |
|  2 |       0.77 |                    5   |                    4.3 |        1    | Astapor        | 2014-01-07 00:00:00 | iPhone  | 2014-01-06 00:00:00 |         0   |                        3 | False             |         100   |
|  3 |       2.36 |                    4.9 |                    4.6 |        1.14 | King's Landing | 2014-06-29 00:00:00 | iPhone  | 2014-01-10 00:00:00 |        20   |                        9 | True              |          80   |
|  5 |      10.56 |                    5   |                    3.5 |        1    | Winterfell     | 2014-06-06 00:00:00 | iPhone  | 2014-01-09 00:00:00 |         0   |                        2 | True              |         100   |

**Table 2**: Initial data type and null value descriptions 

 |   column name |   information | 
 |---:|-----------:|
|avg_dist  |                50000 non-null float64 |
|avg_rating_by_driver   |   49799 non-null float64|
|avg_rating_of_driver  |    41878 non-null float64|
|avg_surge    |             50000 non-null float64|
|city    |                  50000 non-null object|
|last_trip_date    |        50000 non-null object|
|phone       |              49604 non-null object|
|signup_date    |           50000 non-null object|
|surge_pct      |           50000 non-null float64|
|trips_in_first_30_days   | 50000 non-null int64|
|luxury_car_user |          50000 non-null bool|
|weekday_pct    |           50000 non-null float64|

The dataset needed to be cleaned prior to model building. Looking at the Table 2, it is clear that the `avg_rating_by_driver`, `avg_rating_of_driver`, and `phone` columns have null values. These will need to be removed prior to model building.

**Table 3**: Descriptive statistics of dataset summarizing the central tendency, dispersion and shape of distribution

|       |    avg_dist |   avg_rating_by_driver |   avg_rating_of_driver |    avg_surge |   surge_pct |   trips_in_first_30_days |   weekday_pct |
|:------|------------:|-----------------------:|-----------------------:|-------------:|------------:|-------------------------:|--------------:|
| count | 44698       |            44698       |           44698        | 44698        |  44698      |              44698       |    44698      |
| mean  |     5.37761 |                4.82288 |               4.64124  |     1.04483  |      5.9548 |                  1.96834 |       61.3783 |
| std   |     4.32101 |                0.28789 |               0.453777 |     0.102086 |     12.1315 |                  2.43967 |       36.4007 |
| min   |     0       |                3.5     |               3        |     1        |      0      |                  0       |        0      |
| 25%   |     2.44    |                4.7     |               4.5      |     1        |      0      |                  0       |       33.3    |
| 50%   |     3.88    |                5       |               4.8      |     1        |      0      |                  1       |       66.7    |
| 75%   |     6.77    |                5       |               5        |     1.03     |      6.3    |                  3       |      100      |
| max   |    22.91    |                5       |               5        |     1.71     |     66.7    |                 13       |      100      |



**Table 4**: Cleaned data type and null value descriptions 

 |   column name |   information | 
 |---:|-----------:|
|avg_dist  |                50000 non-null float64 |
|avg_rating_by_driver   |   49799 non-null float64|
|avg_rating_of_driver  |    41878 non-null float64|
|avg_surge    |             50000 non-null float64|
|city    |                  50000 non-null object|
|last_trip_date    |        50000 non-null object|
|phone       |              49604 non-null object|
|signup_date    |           50000 non-null object|
|surge_pct      |           50000 non-null float64|
|trips_in_first_30_days   | 50000 non-null int64|
|luxury_car_user |          50000 non-null bool|
|weekday_pct    |           50000 non-null float64|
