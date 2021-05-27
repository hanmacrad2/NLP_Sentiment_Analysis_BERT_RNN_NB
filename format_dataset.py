"""
Script to standardize the initial dataset clean up.

Includes:
* Removing duplicate whitespace
* Removing newline characters
"""
print('Start')
import pandas as pd
import pdb
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

#Initialise
DIRECTORY = 'data/'


scaryLettersMap = {'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
            'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'ª': 'A',
            'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
            'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
            'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
            'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'º': 'O',
            'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
            'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
            'Ñ': 'N', 'ñ': 'n',
            'Ç': 'C', 'ç': 'c',
            '§': 'S',  '³': '3', '²': '2', '¹': '1'}

def clean_review_text(df):
    """
    Parse review text for model training
    """

    # Remove stop words
    s_words = stopwords.words('english')
    s_words_pattern = r'\b(?:{})\b'.format('|'.join(s_words))
    df['text'] = df['text'].str.replace(s_words_pattern, '',
        case=False, regex=True)
    print('\tRemoved stop words')

    # Remove special letter characters
    df['text'] = df['text'].replace(scaryLettersMap, regex=True)
    print('\tRemoved scary letters')

    # Remove <br> and &nbsp;
    # (Special characters are removed later)
    df['text'] = df['text'].str.replace('nbsp', ' ')
    df['text'] = df['text'].str.replace('<br', ' ')
    print('\tRemoved breaking characters')

    # Remove all non letter characters (including punctuation)
    # df['text'] = df['text'].str.replace(string.punctuation, ' ', regex=True)
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('[^a-z]', ' ', regex=True)
    print('\tRemoved non-alphabetical characters')

    # Remove duplicate whitespace
    df['text'] = df['text'].str.replace('\s+', ' ', regex=True)
    print('\tRemoved duplicate whitespace')

    #Replace 1/2 with 0/1
    df['polarity'].replace(1, 0, inplace=True)
    df['polarity'].replace(2, 1, inplace=True)

    print(df.head())

    return df
    
def get_train_test_sets():
    'Format train & test datasets'

    datasets = ['train', 'test']
    # Only use first lines[0] of the training dataset and the first lines[1] of
    # test dataset
    lines = [100000, 20000]

    for i in range (2):
        filepath = DIRECTORY + datasets[i] + '.csv'
        print('Formatting %s dataset' % filepath)

        # First X reviews do not contain the same number of positive & negative
        # reviews. Add a buffer to ensure a 50:50 split in dataset
        buffer_num_lines = int(lines[i] * 1.25)
        df = pd.read_csv(filepath,
            encoding='utf-8',
            nrows=buffer_num_lines,
            names=["polarity", "title", "text"])

        # Group by polarity and get 50:50 split for polarity
        df = df.groupby('polarity').head(lines[i]/2).reset_index(drop=True)

        df.drop(columns=['title'], inplace=True)

        df = clean_review_text(df)

        # Save formatted data to csv
        fmtted_filepath = DIRECTORY + datasets[i] + '_formatted.csv'
        df.to_csv(fmtted_filepath, index=False, header=True)
    
if __name__ == '__main__':
    get_train_test_sets()
