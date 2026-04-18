import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import numpy as np
import random

class ModelTrainer:
    def __init__(self, cfg, seed):
        self.binary_hidden_layer_sizes = (cfg["binary_hidden_layer_sizes"][0],cfg["binary_hidden_layer_sizes"][1],cfg["binary_hidden_layer_sizes"][2])
        self.regressor_hidden_layer_sizes = (cfg["regressor_hidden_layer_sizes"][0],cfg["regressor_hidden_layer_sizes"][1],cfg["regressor_hidden_layer_sizes"][2])
        self.ternary_hidden_layer_sizes = (cfg["ternary_hidden_layer_sizes"][0],cfg["ternary_hidden_layer_sizes"][1],cfg["ternary_hidden_layer_sizes"][2])
        self.clf_activation = cfg["clf_activation"]
        self.reg_activation = cfg["reg_activation"]
        self.model_noise = cfg["model_noise"]
        self.seed = seed

    def get_report(self, y_test, y_pred, classifier = False):

        if classifier:
            y_test = [int(x) for x in y_test]
            y_pred = [int(x) for x in y_pred]
            report = classification_report(y_test, y_pred, output_dict=False)
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred)) 

            report = {
                "R2": r2,
                "MSE": mse,
                "MAE": mae }
        
        return report


    def noisy_output(self, model, X, granularity, flip_rate):
        if granularity[0] == "c":
            return self.noisy_regressor(model, X, flip_rate)
        if granularity[0] == "b":
            return self.noisy_binary(model, X, flip_rate)
        if granularity[0] == "t":
            return self.noisy_ternary(model, X, flip_rate)


    def flip_labels(self, labels, flip_rate, classes):
        """Reassign a fraction of predictions to a wrong class with noise."""

        if self.seed is not None:
            np.random.seed(self.seed)
            
        if random.random() < flip_rate:
            wrong_classes = [c for c in classes if c != labels]
            noisy = np.random.choice(wrong_classes)
        else:
            noisy = labels
        return noisy


    def noisy_binary(self, model, preds, flip_rate=0.1):
        """
        flip_rate=0.1 → ~0.1% of predictions are flipped to the other class.
        Roughly degrades F1 by the flip_rate amount.
        """

        return self.flip_labels(preds, flip_rate, classes=[0, 1])


    def noisy_ternary(self, model, preds, flip_rate=0.1):
        """
        flip_rate=0.1 → ~10% of predictions flipped to one of the other two classes.
        """

        return self.flip_labels(preds, flip_rate, classes=[0, 1, 2])


    def noisy_regressor(self, model, preds, noise_level=0.1):
        """
        noise_level=0.1 → noise std = 10% of the prediction's own std.
        Degrades R² roughly proportionally.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        noise = np.random.normal(0, noise_level, size=preds.shape)
        return preds + noise

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
                        hidden_layer_sizes=self.binary_hidden_layer_sizes,      #10 5 2
                        activation='relu',
                        solver='adam', 
                        early_stopping=False,  
                        max_iter = 300,           
                        random_state=random_state)

        if granularity == "discrete" or granularity == "ternary":
            clf = MLPClassifier(
                        hidden_layer_sizes=self.ternary_hidden_layer_sizes,      
                        activation='relu',
                        solver='adam', 
                        early_stopping=False,  
                        max_iter = 300, 
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
                        hidden_layer_sizes=self.regressor_hidden_layer_sizes,      
                        activation='tanh',
                        solver='adam', 
                        early_stopping=False,  
                        max_iter = 300, 
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