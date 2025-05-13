from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


class GridSearch:

    def define_params(self, n_estimators_list, max_depth_list, min_samples_split_list):
        self.param_grid = {
            'n_estimators': n_estimators_list,
            'max_depth': max_depth_list,
            'min_samples_split': min_samples_split_list
        }

    def create_grid_search(self):
        self.grid_search = GridSearchCV(
            estimator = RandomForestRegressor(),
            param_grid = self.param_grid,
            cv=3,
            scoring='neg_mean_absolute_error', 
            n_jobs=-1, 
            verbose=2 
        )

    def create_pipeline(self, preprocessor):
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', self.grid_search)
        ])