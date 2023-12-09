import os
import pandas as pd

def get_data_frame(file):

    # current directory
    current_dir = os.path.dirname('src')

    # directory of data relative to current directory
    data_file_path = os.path.join(current_dir, '..', 'data', file)

    # read the csv file
    df = pd.read_csv(data_file_path)

    # standardise hr names across databases
    df = standard_hr_column(df)

    return df

def standard_hr_column(df):

    # formats the hr name: HR9110
    df['name'] = df['name'].str.strip()
    df['name'] = df['name'].str.replace(' ', '')

    return df
