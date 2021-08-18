from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydataset import data
import statistics
import acquire
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")



def clean_iris(df):
    df = df.drop(columns = ['species_id'])
    df = df.rename(columns={"species_name": "species"})
    df_dummy = pd.get_dummies(df['species'], drop_first=True)
    df= pd.concat([df, df_dummy], axis = 1)
    return df

def split_iris_data(df):
    """
    splits the data in train validate and test 
    """
    train, test = train_test_split(df, test_size = 0.2, random_state = 123, stratify = df.species)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.species)
    
    return train, validate, test

def prep_irs_data(df):
    """
    takes in a data from titanic database, cleans the data, splits the data
    in train validate test and imputes the missing values for embark_town. 
    Returns three dataframes train, validate and test.
    """
    df = clean_iris(df)
    train, validate, test = split_iris_data(df)
    #train, validate, test = impute_mode(train, validate, test) #nothing to impute
    return train, validate, test

def prep_telco():
    df = acquire.get_telco_data()
    df = df.drop_duplicates()
    df = df.drop(columns = ['customer_id'])
    df_dummy = pd.get_dummies(df[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing'  ]], drop_first=[True, True, True, True, True, True, True, True, True, True, True, True])
    df= pd.concat([df, df_dummy], axis = 1)
    return df