from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

class MlModel:
    def create_pipline(self, preprocessor):
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=10))
        ])

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        return self.pipeline.predict(X_test)