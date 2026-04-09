import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import numpy as np

class ModelTrainer:
    def create_train_data(self, df, random_state = 42):
    
        min_discrete = int(df['discrete_optimal'].value_counts().min())
        df_balanced_discrete = df.groupby('discrete_optimal', group_keys=False).apply(lambda x: x.sample(min_discrete, random_state=random_state, replace=True))

        # Downsample binary labels
        min_binary = int(df['binary_optimal'].value_counts().min())
        # print("Minimum count for binary_label:", min_binary)
        df_balanced_binary = df.groupby('binary_optimal', group_keys=False).apply(lambda x: x.sample(min_binary, random_state=random_state, replace=True))

        # Separate features and labels
        X_binary = df_balanced_binary[df.columns]
        X_discrete = df_balanced_discrete[df.columns]
        X_continuous = df[df.columns]

        y_discrete = df_balanced_discrete['discrete_optimal']
        y_binary = df_balanced_binary['binary_optimal']
        y_continuous = df['continuous_optimal']

        return X_continuous, X_binary, X_discrete, y_binary, y_discrete, y_continuous

    def train_classifier(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.1, random_state: int = 42, shuffle_data: bool = True):
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle_data, stratify=y)
        clf = MLPClassifier(
                    hidden_layer_sizes=(15, 6, 2),      
                    activation='relu',
                    solver='adam',              
                    random_state=random_state)
        clf.fit(X, y)

        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=False)
        
        return clf, report
    
    def train_regressor(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.1, random_state: int = 42, shuffle_data: bool = True):
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle_data, stratify=y)

        clf = MLPRegressor(max_iter=1000)
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