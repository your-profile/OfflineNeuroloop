from collections import deque
from turtle import pd
import pandas as pd
import numpy as np

class fNIRSBuffer:
    def __init__(self, window_duration_s: float = 60.0, sample_period_s: float = 10.0, neural_channels = ["L_O_DSI", "L_D_DSI", "L_O_DSphi", "L_D_DSphi", "R_O_DSI", "R_D_DSI", "R_O_DSphi", "R_D_DSphi"], verbose: bool = False):
        self.window_duration_s = window_duration_s
        self.sample_period_s = sample_period_s
        self.channels = neural_channels
        self.max_len = 500
        self.timestamps: deque = deque(maxlen=self.max_len) # timestamps
        self.indices: deque = deque(maxlen=self.max_len) # indices
        self.values: deque = deque(maxlen=self.max_len) # fnirs channel inputs
        self.classifications: deque = deque(maxlen=self.max_len) # model classifications
        self.curr_length = -1
        self.verbose = verbose

    def add_sample(self, timestamp: pd.Timestamp, x: np.ndarray, classification: float) -> None:
        """
        Takes in a row from the aligned fNIRS dataset. Adds the index, neural data and timestamp at that time, as long as the index is unique (unseen)
        """

        if self.curr_length < 0 or self.indices[self.curr_length] < x.name:
            self.timestamps.append(timestamp)
            self.values.append(np.asarray(x[self.channels], dtype=float))
            self.indices.append(x.name)
            self.classifications.append(classification)

            if self.curr_length < self.max_len -1 :
                self.curr_length += 1   

    def get_neural_credit(self, X: int = 5):
        """
        Looks at the last X classifications and uses majority voting to determine the classification.
        Returns the majority class (float or int).
        If there is a tie, returns the most recent one among the tied classes.
        """
        if len(self.classifications) < X:
            return 0.0

        relevant_classes = list(self.classifications)[-X:]

        if len(relevant_classes) == 0:
            return 0.0

        classes, counts = np.unique(np.array(relevant_classes), return_counts=True)
        max_count = classes[np.argmax(counts)]

        # print(f"Classifications: {counts}, Final Class: {max_count}")

        return int(max_count)

         

    def get_window(self):
        # TODO: Adding window length

        window_steps = int(round(self.window_duration_s / self.sample_period_s))
        starting_idx = len(self.timestamps) - window_steps
        # print(f"Window Steps: {window_steps}")


        if not self.values or starting_idx < 0:
            if self.verbose:
                print(f"Window Length Necessary: {window_steps} -- At Starting Index: {starting_idx}")
            return None, None

        ts = np.array(self.timestamps)
        ts = ts[starting_idx:]
        X = np.stack(self.values, axis=0)  # [T, C] time x channels
        X = X[starting_idx:]

        return ts, X

    def get_features(self): #add other features
        ts, X = self.get_window()
        if X is None or len(X) < 2:
            return None

        mean = X.mean(axis=0)
        std = X.std(axis=0)

        t = np.arange(len(X))[:, None]
        t_centered = t - t.mean()
        denom = (t_centered ** 2).sum()

        if denom == 0:
            slope = np.zeros(X.shape[1])
            intercept = mean  # fallback to mean as intercept if no change
        else:
            slope = (t_centered * (X - X.mean(axis=0))).sum(axis=0) / denom
            intercept = X.mean(axis=0) - slope * t.mean()

        # Skewness and kurtosis per channel
        from scipy.stats import skew, kurtosis
        skewness = skew(X, axis=0, bias=False)
        kurt = kurtosis(X, axis=0, fisher=True, bias=False)

        # combine features
        features = np.concatenate([mean, std, slope, intercept, skewness, kurt], axis=0)
        return features