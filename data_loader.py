import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

class DataLoader:

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        return pd.read_parquet(self.filepath, engine='fastparquet')

    def clean(slef, df):
        df.dropna(how='any', axis=0, inplace=True)
    
    def tripDuration(self, df):
        df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
        df.drop('tpep_pickup_datetime', axis=1, inplace=True)
        df.drop('tpep_dropoff_datetime', axis=1, inplace=True)

    def split(self, df):
        target_variable=df['total_amount']
        df.drop('total_amount', axis=1,  inplace=True)
        return train_test_split(df, target_variable, test_size=0.2)
    
    def baselineList(self, y_train, y_test):
        mean = y_train.mean()
        return [mean] * len(y_test)
    
    def preprocess(self, df):
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        continuous_columns = df.select_dtypes(include=['number']).columns.tolist()

        preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), continuous_columns),  
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
            ])
        
        return preprocessor
    
