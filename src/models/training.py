import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import numpy as np

class ModelTrainer:
    def train_classifier(self, 
                         X: np.ndarray, 
                         y: np.ndarray, 
                         test_size: float = 0.1, 
                         granularity = "binary", 
                         random_state: int = 42, 
                         shuffle_data: bool = True):
        if granularity == "continuous":
            return self.train_regressor(X, y, test_size, random_state, shuffle_data)
        
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, 
                                                random_state=random_state, 
                                                shuffle=shuffle_data, 
                                                stratify=y)
        
        if granularity == "binary":
            clf = MLPClassifier(
                        hidden_layer_sizes=(10, 5, 2),      
                        activation='relu',
                        learning_rate=0.001,             
                        solver='adam',              
                        random_state=random_state)

        if granularity == "discrete" or granularity == "ternary":
            clf = MLPClassifier(
                        hidden_layer_sizes=(18, 6, 3),      
                        activation='relu',
                        solver='adam', 
                        random_state=random_state)

        clf.fit(X, y)

        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=False)
        
        return clf, report
        
    def train_regressor(self, 
                        X: np.ndarray, 
                        y: np.ndarray, 
                        test_size: float = 0.1, 
                        random_state: int = 42, 
                        shuffle_data: bool = True):

        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, 
                                                random_state=random_state, 
                                                shuffle=shuffle_data)

        clf = MLPRegressor(
                        hidden_layer_sizes=(20, 5, 3),      
                        activation='tanh',
                        solver='adam', 
                        random_state=random_state)
                        
        clf.fit(X, y)

        y_pred = clf.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred)) 

        report = {
            "R2": r2,
            "MSE": mse,
            "MAE": mae
        }

        return clf, report