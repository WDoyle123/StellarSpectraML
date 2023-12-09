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
    df = standardise_column(df, 'name')

    return df

def standardise_column(df, column):

    # formats the hr name: HR9110
    df[column] = df[column].str.strip()
    df[column] = df[column].str.replace(' ', '')

    return df

def query_simbad():
    '''
    This function is used to query the SIMBAD database
    We query for each HR name to return the HR name and the spectral type
    '''
    df = get_data_frame('ybs_data.csv')

    df.name = df.name.str.replace('HR', '')

    with open('../data/query_simbad.txt', 'w') as file:
        file.write('format object f1 "%IDLIST(HR), %SP(S)""\n')
        for row in df.name:
            file.write(f'query id HR {row}\n')

def combined_data_frame():

    # get Yale Bright Star Catalogue
    ybs_df = get_data_frame('ybs_data.csv')

    # get data from the SIMBAD query
    spectral_df = get_data_frame('spectral_type.csv')
    
    # prep dataframe for join
    spectral_df = standardise_column(spectral_df, 'spectral_type')
    
    # Grab simple spectral type (just the letter)
    spectral_df.spectral_type = spectral_df.spectral_type.str[0]

    # spectral types we want to use
    standard_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'S', 'L', 'D']
    spectral_df = spectral_df[spectral_df['spectral_type'].isin(standard_types)]

    # merge the dataframes
    combined_df = pd.merge(ybs_df, spectral_df, on='name', how='inner', suffixes=('_ybs', '_simbad'))

    # save the data frame (make it easy to examime the csv)
    combined_df.to_csv('../data/combined_df.csv')
    
    return combined_df


if __name__ == '__main__':
    query_simbad()
