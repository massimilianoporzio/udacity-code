import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    #Convert relevant parts of ndc_df to a dictionary
    ndc_df.set_index('NDC_Code', inplace=True)
    ndc_dict = ndc_df['Non-proprietary Name'].to_dict()
    
    #Map original dataframe codes to generic names
    df['generic_drug_name'] = df['ndc_code']
    df['generic_drug_name'] = df['generic_drug_name'].map(ndc_dict, na_action='ignore')
    return df

#Question 4

def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    
    first_encounter_df = df.copy()
    #sort based on encounter_id
    first_encounter_df = first_encounter_df.sort_values('encounter_id')
    #grouping by patient number end selecting the firts encounter id (after sorting)
    first_encounter_values = df.groupby('patient_nbr')['encounter_id'].head(1).values
    #selecting only th records in the first_encounter_values
    first_encounter_df = first_encounter_df[first_encounter_df['encounter_id'].isin(first_encounter_values)]
    #dropping duplicates and retaining one (the first) encounter
    first_encounter_df.drop_duplicates(subset=['patient_nbr'], inplace=True)
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id
    

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    train_percentage=0.6
    #shuffling
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    sample_size = round(total_values * (train_percentage ))
    train = df[df[patient_key].isin(unique_values[:sample_size])].reset_index(drop=True)
    bucket = df[df[patient_key].isin(unique_values[sample_size:])].reset_index(drop=True)
    #NOW SPLITTING KEEPING UNIQUE PATIENTS WITHOUT LEAKAGE ALSO FOR VALIDATION / TEST DATASETS
    unique_values_bucket = bucket[patient_key].unique()
    total_values_bucket = len(unique_values_bucket)
    sample_size_bucket = total_values_bucket // 2
    validation = bucket[bucket[patient_key].isin(unique_values_bucket[:sample_size_bucket])].reset_index(drop=True)
    test = bucket[bucket[patient_key].isin(unique_values_bucket[sample_size_bucket:])].reset_index(drop=True)
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
         key=c,vocabulary_file = vocab_file_path, num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)        
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    tf_numeric_feature = tf.feature_column.numeric_column(
                    key=col,  dtype=tf.float64,
                    normalizer_fn= lambda x: normalize_numeric_with_zscore(x,MEAN,STD))
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = '?'
    s = '?'
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    return student_binary_prediction
