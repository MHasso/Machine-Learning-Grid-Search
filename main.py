
from data_loader import DataLoader
from ml_model import MlModel
from grid_search import GridSearch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline



from sklearn.metrics import mean_absolute_error


def main():
    filename = 'yellow_tripdata.parquet'
    dataloader = DataLoader(filename)

    #load data
    df = dataloader.load()

    #clean data
    dataloader.clean(df)

    # create new trip duration coilumn
    dataloader.tripDuration(df)

    #split data
    X_train, X_test, y_train, y_test = dataloader.split(df)
    print(X_train)

    #create a baseline mean for comparison
    y_pred_baseline_list = dataloader.baselineList(y_train, y_test)

    # print the baseline error
    print(mean_absolute_error(y_test, y_pred_baseline_list))

    #transform categorical data and continous features
    preprocessor = dataloader.preprocess(df)

    # create the model pipeline
    mlmodel = MlModel()
    mlmodel.create_pipline(preprocessor)
    #mlmodel.fit(X_train, y_train)
    #y_pred = mlmodel.predict(X_test)
    #mae = mean_absolute_error(y_test, y_pred)
    #print(mae)

    # create grid search to search for the best parameters for the random forest regressor
    gridsearch = GridSearch()
    gridsearch.define_params([50, 100, 200], [10, 20, 30], [2, 5, 10])
    gridsearch.create_grid_search()
    gridsearch.create_pipeline(preprocessor)
    #gridsearch.pipeline.fit(X_train, y_train)

    #print the best model and parameters 
    #print(gridsearch.grid_search.best_estimator_)
    #print(gridsearch.grid_search.best_score_)
    #print(gridsearch.grid_search.best_params_)

    # Fit the best classifier on the training data.
    pipeline_RandomForestRegressor = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(max_depth=30, n_estimators=200, min_samples_split=2))
        ])
    pipeline_RandomForestRegressor.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = pipeline_RandomForestRegressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(mae) 

if __name__ == "__main__":
     main()