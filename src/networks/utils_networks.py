import os
import pickle
import pandas as pd
import numpy as np

class Data():  
    '''
    Data: Class for utility functions to manipulate data. Reading files, creating labels etc...
    '''

    def read_files(participant_list, source_folder_1, conditions):
        participant_data = {}

        for filename in os.listdir(source_folder_1):
            file_path = os.path.join(source_folder_1, filename)

            for participant in participant_list:
                participant_str = f"{participant:03}"  #always 3 nums

                if filename.startswith(participant_str):
                    for condition in conditions:
                        # Check if the condition is in the filename
                        if condition in filename:
                            # Load pickle or csv
                            if filename.endswith('.pickle'):
                                with open(file_path, 'rb') as f:
                                    df = pickle.load(f)
                            elif filename.endswith('.csv'):
                                df = pd.read_csv(file_path, index_col=0)
                            else:
                                continue

                            key = f"{participant_str}{condition}"
                            participant_data[key] = df

        return participant_data
    @classmethod
    def from_tensor(self, x):
        try:
            x = x[0].detach().cpu().numpy()
            return self.softmax(x)
        except:
            return x   
        
    def apply_features(combined_df, channels):
        for channel in channels:
            combined_df[channel] = combined_df[channel].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

            max_length = max(combined_df[channel].apply(len))

            feature_columns = pd.DataFrame(combined_df[channel].tolist(), columns=[f'{channel}_{i}' for i in range(max_length)])
            combined_df = pd.concat([combined_df, feature_columns], axis=1).drop(columns=[channel])
        return combined_df
        
    def softmax(values):
        values = np.asarray(values, dtype=np.longdouble)
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return np.asarray(exp_values / np.sum(exp_values), dtype=np.float64)

    def cosine_similarity(vector1, vector2):
        if vector1 is None or vector2 is None:
            return 0
        
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


    def KL_regression_labels(Q, P):
        from scipy.special import rel_entr

        return sum(rel_entr(P, Q))

    def CE_regression_labels(O, P):

        cross_entropy = -np.sum(O * np.log(P + 1e-9))

        return cross_entropy

    def euclideanDist(vector1, vector2):
        if vector1 is None or vector2 is None:
            return 0
        
        return np.linalg.norm(vector1 - vector2)

    def MSE(vector1, vector2):
        if vector1 is None or vector2 is None:
            return 0
        
        return np.mean((vector1 - vector2) ** 2)
