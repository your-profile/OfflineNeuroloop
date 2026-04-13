import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import numpy as np

class ModelTrainer:

    def noisy_output(self, model, X, granularity, flip_rate, seed=None):
        if granularity[0] == "c":
            return self.noisy_regressor(model, X, flip_rate, seed=None)
        if granularity[0] == "b":
            return self.noisy_binary(model, X, flip_rate, seed=None)
        if granularity[0] == "c":
            return self.noisy_ternary(model, X, flip_rate, seed=None)


    def flip_labels(self, labels, flip_rate, classes, seed=None):
        """Reassign a fraction of predictions to a wrong class with noise."""

        if seed is not None:
            np.random.seed(seed)
        
        noisy = labels.copy()
        n = len(labels)
        n_flip = int(n * flip_rate)
        flip_idx = np.random.choice(n, size=n_flip, replace=False)
        
        for i in flip_idx:
            wrong_classes = [c for c in classes if c != labels[i]]
            noisy[i] = np.random.choice(wrong_classes)
        
        return noisy


    def noisy_binary(self, model, preds, flip_rate=0.1, seed=None):
        """
        flip_rate=0.1 → ~0.1% of predictions are flipped to the other class.
        Roughly degrades F1 by the flip_rate amount.
        """
        return self.flip_labels(preds, flip_rate, classes=[0, 1], seed=seed)


    def noisy_ternary(self, model, preds, flip_rate=0.1, seed=None):
        """
        flip_rate=0.1 → ~10% of predictions flipped to one of the other two classes.
        """
        return self.flip_labels(preds, flip_rate, classes=[0, 1, 2], seed=seed)


    def noisy_regressor(self, model, preds, noise_level=0.1, seed=None):
        """
        noise_level=0.1 → noise std = 10% of the prediction's own std.
        Degrades R² roughly proportionally.
        """
        if seed is not None:
            np.random.seed(seed)
        
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
                        hidden_layer_sizes=(10, 5, 2),      
                        activation='relu',
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