import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

path = r'E:\OneDrive - National University of Singapore\2023-2024 SEM 2\BT5153 Applied Machine Learning for Business Analytics\kaggle competition'

def getWhaleDate():
    profile_df = pd.read_csv(path + r'\profile.csv')
    train_label_df = pd.read_csv(path + r'\train_label.csv')
    trx_df = pd.read_csv(path + r'\trx_data.csv')
    trx_df['transaction_time'] = pd.to_datetime(trx_df['transaction_time'])
    trx_df['month'] = trx_df['transaction_time'].dt.month
    trx_df['day'] = trx_df['transaction_time'].dt.day
    trx_df['hour'] = trx_df['transaction_time'].dt.hour
    trx_df['day_of_week'] = trx_df['transaction_time'].dt.day_name()
    deviation = trx_df.groupby(['user_id'])['gtv'].std().fillna(0)

    df1 = trx_df.groupby(['user_id', 'transaction_type', 'month'])['gtv'].count().unstack(['transaction_type', 'month']).fillna(0)
    df1.columns = ['count_'+'_'.join([str(i) for i in col]) for col in df1.columns]

    df2 = trx_df.groupby(['user_id', 'transaction_type', 'month'])['gtv'].sum().unstack(['transaction_type', 'month']).fillna(0)
    df2.columns = ['sum_'+'_'.join([str(i) for i in col]) for col in df2.columns]

    df = pd.concat([df1, df2], axis=1)
    df = df.merge(trx_df.groupby(['user_id', 'asset_type'])['gtv'].sum().unstack('asset_type').fillna(0), on='user_id', how='left')
    df['std'] = deviation

    df['sum_SUM_4'] = df['sum_SELL_4'] + df['sum_BUY_4']
    df['sum_SUM_5'] = df['sum_SELL_5'] + df['sum_BUY_5']
    df['sum_DIFF_4'] = df['sum_SELL_4'] - df['sum_BUY_4']
    df['sum_DIFF_5'] = df['sum_SELL_5'] - df['sum_BUY_5']
    df['sum_INC'] = (df['sum_SUM_5'] / df['sum_SUM_4']).replace([float('inf'), -float('inf'),np.nan], 0)
    df['sum_SUM_agg'] = df['sum_SUM_4'] + df['sum_SUM_5']
    df['sum_DIFF_agg'] = df['sum_DIFF_4'] + df['sum_DIFF_5']

    df['count_SUM_4'] = df['count_SELL_4'] + df['count_BUY_4']
    df['count_SUM_5'] = df['count_SELL_5'] + df['count_BUY_5']
    df['count_DIFF_4'] = df['count_SELL_4'] - df['count_BUY_4']
    df['count_DIFF_5'] = df['count_SELL_5'] - df['count_BUY_5']
    df['count_INC'] = (df['count_SUM_5'] / df['count_SUM_4']).replace([float('inf'), -float('inf'),np.nan], 0)
    df['count_SUM_agg'] = df['count_SUM_4'] + df['count_SUM_5']
    df['count_DIFF_agg'] = df['count_DIFF_4'] + df['count_DIFF_5']

    # Extract date from 'transaction_time'
    trx_df['date'] = trx_df['transaction_time'].dt.date

    # Count unique dates for each id
    unique_days_per_id = trx_df.groupby('user_id')['date'].nunique()

    df['trx_days'] = unique_days_per_id

    # Find the last transaction day for each id
    last_transaction_day = trx_df.groupby('user_id')['transaction_time'].max()

    # Calculate the number of days between the last transaction day and 2022-06-01
    days_to_2022_06_01 = (pd.to_datetime('2022-06-01') - last_transaction_day).dt.days

    df['trx_gap'] = days_to_2022_06_01

    combined_df = df.merge(profile_df, left_index=True, right_on='user_id')
    combined_df.set_index('user_id', inplace=True)
    # combined_df = combined_df.drop('user_id',axis= 1)

    # Merge combined_df with train_label_df on user_id
    train_df = combined_df.merge(train_label_df, on='user_id')
    train_df = train_df.drop('user_id', axis=1)

    # Create test_df by excluding the user_ids present in train_df
    test_df = combined_df[~combined_df.index.isin(train_label_df['user_id'])]
    # test_df = test_df.drop('user_id', axis=1)

    # print(train_df.shape, test_df.shape)
    # print(len(train_df) /(len(train_df)+ len(test_df)))
    train_df_shuffle = train_df.groupby(['tgt']).apply(lambda x: x.sample(frac=0.1,replace=False,random_state=1)).sample(frac=1, random_state=1, replace= False ).reset_index(drop=True)
    y = train_df_shuffle['tgt']

    # ['BUY_4', 'BUY_5', 'SELL_5', 'SELL_4', 'SUM_4', 'SUM_5', 'DIFF_4',
    #        'DIFF_5', 'INC', 'mobile_brand_name', 'mobile_marketing_name',
    #        'age_in_year', 'gender_name', 'marital_status', 'education_background',
    #        'income_level', 'occupation', 'tgt']

    # Define columns for each type of encoder:
    numerical_columns = ['count_BUY_4', 'count_BUY_5', 'count_SELL_5', 'count_SELL_4',
        'sum_BUY_4', 'sum_BUY_5', 'sum_SELL_5', 'sum_SELL_4', 'sum_SUM_4',
        'sum_SUM_5', 'sum_DIFF_4', 'sum_DIFF_5', 'sum_INC', 'count_SUM_4',
        'count_SUM_5', 'count_DIFF_4', 'count_DIFF_5', 'count_INC', 'sum_SUM_agg' , 
        'sum_DIFF_agg', 'count_SUM_agg', 'count_DIFF_agg',
        'crypto', 'fx','trx_days', 'trx_gap',
        'gold', 'gss', 'idss', 'mfund', 'stock_index','std',
        'age_in_year']  # numerical columns
    ordinal_columns = ['education_background', 'income_level']     # categorical columns for ordinal encoding
    onehot_columns = [ 'gender_name', 'marital_status', 'occupation'] # categorical columns for onehot encoding
    onehot_pca_columns = ['mobile_brand_name', 'mobile_marketing_name'] # categorical columns for onehot encoding with PCA

    # Define the encoders
    ordinal = OrdinalEncoder(categories = [
                                ['others', 'primary school','junior high school', 'senior high school', 
                                            'diploma','undegraduate', 'post graduate'], # Savings account
                                ['< 10 million/year','> 10 – 50 million/year', 
                                '> 50 – 100 million/year','> 100 – 500 million/year', '> 500 million – 1 billion/year',]],
                            handle_unknown = 'use_encoded_value',  
                            unknown_value = -1
                            )

    onehot = OneHotEncoder(categories = [
                                ['Male', 'Female',np.nan], # Sex
                                [ 'single', 'married', 'divorced',np.nan], # Housing
                                [ 'private employee', 'others', 'student', 'housewife', 'civil servant','lecturer/teacher', 
                                'indonesian national armed force/indonesian national police','retired',np.nan], # Occupation
                            ],
                            handle_unknown = 'error',
                            drop='first' # to return k-1, use drop=False to return k
                            )
    scaler = StandardScaler()
    onehot_pca = Pipeline([
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
        ('pca', PCA(n_components=10))
    ])

    # Combine them!
    preprocessor = ColumnTransformer([
    #(nickname, transformer to apply, columns to apply to)   
        ('numerical', 'passthrough', numerical_columns),   # <- 'passthrough' says to keep them but don't apply anything
        ('ordinal', ordinal, ordinal_columns),             # <- apply ordinal encoder to the ordinal_columns
        ('onehot', onehot, onehot_columns),                 # <- apply onehot encoder to the onehot_columns
        ('onehot_pca', onehot_pca, onehot_pca_columns)                 # <- apply onehot encoder to the onehot_columns
    ])

    train_df_shuffle = pd.DataFrame(
        data = preprocessor.fit_transform(train_df_shuffle),
        columns = preprocessor.get_feature_names_out(), # <- get the encoded feature names
        index = train_df_shuffle.index
    )
    return train_df_shuffle, y