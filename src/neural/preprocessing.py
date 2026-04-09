import pandas as pd
import numpy as np
from typing import Tuple, List

class DatasetProcessor:
    def __init__(self):
        self.neural_channels = None
        self.verbose = True
        self.gap_threshold_s = 600
        self.fnirs_df = None

    def align_streams(self, fnirs_df: pd.DataFrame, task_df: pd.DataFrame, labels_df: pd.DataFrame, resample_rate_hz: float = 5.2, label_tolerance_s: float = 0.3, neural_channels: List[str] = ["L_O_DSI", "L_D_DSI", "L_O_DSphi", "L_D_DSphi", "R_O_DSI", "R_D_DSI", "R_O_DSphi", "R_D_DSphi"], gap_threshold_s: float = 600, verbose = False) -> Tuple[pd.DataFrame, List[str]]:

        if self.neural_channels == None:
            self.neural_channels = neural_channels
        
        # compute sampling period
        period_ms = int(1000.0 / resample_rate_hz)
        rule = f"{period_ms}ms"

        fnirs_df = fnirs_df.copy()
        task_df = task_df.copy()
        labels_df = labels_df.copy()

        fnirs_df['time'] = pd.to_datetime(fnirs_df['time'], utc=True)
        task_df['time'] = pd.to_datetime(task_df['time'], utc=True)
        labels_df['time'] = pd.to_datetime(labels_df['time'], utc=True)

        fnirs_df = fnirs_df.sort_values('time').set_index('time')
        task_df = task_df.sort_values('time').set_index('time')
        labels_df = labels_df.sort_values('time').set_index('time')

        # choose fnirs channels
        fnirs_channels = [c for c in fnirs_df.columns if c in neural_channels]
        fnirs_numeric = fnirs_df[fnirs_channels]

        # split fnirs based on large time gaps
        gap = fnirs_numeric.index.to_series().diff().dt.total_seconds()
        segment_id = (gap > gap_threshold_s).cumsum()

        aligned_segments = []

        for seg in segment_id.unique():

            fnirs_seg = fnirs_numeric[segment_id == seg]

            start = fnirs_seg.index.min()
            end = fnirs_seg.index.max()

            # restrict task and labels to this time window only
            task_seg = task_df.loc[start:end]
            labels_seg = labels_df.loc[start:end]

            # resample fnirs
            fnirs_resampled = (fnirs_seg.resample(rule).mean().interpolate())

            task_resampled = task_seg.resample(rule).ffill()

            labels_resampled = labels_seg.reindex(fnirs_resampled.index, method='nearest', tolerance=pd.Timedelta(seconds=label_tolerance_s))
            labels_resampled = labels_resampled.ffill()

            aligned = fnirs_resampled.join(task_resampled, how='left')
            aligned = aligned.join(labels_resampled, how='left', rsuffix='_label')
            aligned_segments.append(aligned)

            if self.verbose or verbose:
                print(len(fnirs_resampled), len(task_resampled), len(labels_resampled))
                # print("Task Segment End: Reward", task_seg["rewards"].iloc[-1])
          

        # concat
        aligned = pd.concat(aligned_segments).sort_index()
        aligned.index.name = 'time'

        return aligned, fnirs_channels

    def build_balanced_binary_dataset(
        self, aligned_df: pd.DataFrame,
        fnirs_channels: List[str],
        label_col: str = 'binary_label_shifted',
        window_duration_s: float = 6.0,
        resample_rate_hz: float = 10.0,
        random_state: int | None = None,
        granularity = "binary"
    ) -> tuple[np.ndarray, np.ndarray]:
        import statistics

        """Create a class-balanced (undersampled) binary dataset.

        This uses the same windowing logic as `build_supervised_dataset`, but
        returns a subset of windows so that the two binary classes have equal size
        (or as close as possible if counts differ slightly).
        """

        X, y_binary, y_discrete, y_continuous = self.build_supervised_dataset(
            aligned_df=aligned_df,
            fnirs_channels=fnirs_channels,
            label_col=label_col,
            window_duration_s=window_duration_s,
            resample_rate_hz=resample_rate_hz,
        )

        rng = np.random.default_rng(random_state)


        # First build full multi-label dataset
        if granularity[0] == "d":
            y = y_discrete
            classes, counts = np.unique(y, return_counts=True)
            min_n = counts.min()
            med_n = statistics.median(counts)
            target_n = (min_n+med_n)/2
            idx_class0 = np.where(y == classes[0])[0]
            idx_class1 = np.where(y == classes[1])[0]
            idx_class2 = np.where(y == classes[2])[0]

        if granularity[0] == "b":
            y = y_binary
            classes, counts = np.unique(y, return_counts=True)
            target_n = counts.min()
            # target_n = (counts.min() + counts.max())//3
            idx_class0 = np.where(y == classes[0])[0]
            idx_class1 = np.where(y == classes[1])[0]

        if granularity[0] == "c":
            y = y_continuous
            percentiles = np.percentile(y, [0, 33.333, 66.666, 100])
            y_binned = np.digitize(y, percentiles[1:-1], right=True)
            classes = np.array([0,1,2])
            #count values in each bucket
            counts = np.array([(y_binned == i).sum() for i in range(3)])


        if (granularity[0] == "b" and len(classes) != 2):
            raise ValueError(f"Expected exactly 2 classes for binary label, got {classes}.")
        if (granularity[0] == "d" and len(classes) != 3):
            raise ValueError(f"Expected exactly 3 classes for discrete label, got {classes}.")


        if len(idx_class0) < target_n:
            class0_n = min(len(idx_class0), len(idx_class1))
        else:
            class0_n = target_n
        if len(idx_class1) < target_n:
            class1_n = min(len(idx_class0), len(idx_class1))
        else:
            class1_n = target_n
        if granularity[0] == "d" and (len(idx_class2) < target_n):
            class2_n = min(len(idx_class0), len(idx_class1), len(idx_class2))
        else:
            class2_n = target_n



        select0 = rng.choice(idx_class0, size=class0_n, replace=False)
        select1 = rng.choice(idx_class1, size=class1_n, replace=False)

        if granularity[0] != "b":
            select2 = rng.choice(idx_class2, size=class2_n, replace=False)
            sel = np.concatenate([select0, select1, select2])
        else:
            sel = np.concatenate([select0, select1])

        rng.shuffle(sel)

        X_bal = X[sel]
        y_bal = y[sel]

        return X_bal, y_bal

    def shift_labels_for_delay(
        self, aligned_df: pd.DataFrame,
        delay_s: float,
        label_col_binary: str = 'binary_optimal',
        label_col_discrete: str = 'discrete_optimal',
        label_col_continuous: str = 'continuous_optimal',
        verbose: bool = False) -> pd.DataFrame:
        df = aligned_df.copy()
        dt = (df.index[1] - df.index[0]).total_seconds()
        shift_periods = int(round(delay_s / dt))

        # detect segment boundaries
        gaps = df.index.to_series().diff().dt.total_seconds()
        segment_id = (gaps > self.gap_threshold_s).cumsum()

        for col_shifted, col_orig in [
            ('binary_label_shifted',   label_col_binary),
            ('discrete_label_shifted', label_col_discrete),
            ('continuous_label_shifted', label_col_continuous),
        ]:
            df[col_shifted] = np.nan
            for seg in segment_id.unique():
                mask = segment_id == seg
                seg_labels = df.loc[mask, col_orig]
                df.loc[mask, col_shifted] = seg_labels.shift(-shift_periods).values

        # drop rows at the end of each segment that have no valid future label
        df = df.dropna(subset=['binary_label_shifted', 'discrete_label_shifted', 'continuous_label_shifted'])

        self.fnirs_df = df
        return df

    def get_fnirs_sample(self,
        timestamp,
        temporal_shift: float = 4.0,
        fnirs_channels: List[str] = ["L_O_DSI", "L_D_DSI", "L_O_DSphi", "L_D_DSphi",
                                    "R_O_DSI", "R_D_DSI", "R_O_DSphi", "R_D_DSphi"]
    ):
        # Convert temporal_shift to Timedelta if it isn't already
        if not isinstance(temporal_shift, pd.Timedelta):
            temporal_shift = pd.Timedelta(seconds=temporal_shift)

        # Find closest fNIRS sample to (timestamp + temporal_shift)
        target_time = timestamp + temporal_shift

        try:
            closest_idx = (self.fnirs_df['time'] - target_time).abs().argmin()
        except:
            closest_idx = (self.fnirs_df.index.to_series() - target_time).abs().argmin()

        fnirs_neural_data = self.fnirs_df.iloc[closest_idx][fnirs_channels]

        return fnirs_neural_data
    # def shift_labels_for_delay(
    #     self, aligned_df: pd.DataFrame,
    #     delay_s: float,
    #     label_col_binary: str = 'binary_optimal',
    #     label_col_discrete: str = 'discrete_optimal',
    #     label_col_continuous: str = 'continuous_optimal',
    #     verbose: bool = False) -> pd.DataFrame:

    #     """Shift labels backward in time to account for a fixed delay.

    #     For example, with delay_s=6, a label observed at t is treated as referring
    #     to neural activity around t-6 seconds.
    #     """

    #     df = aligned_df.copy()

    #     # Convert time delay to number of rows using the DataFrame's sampling rate
    #     dt = (df.index[1] - df.index[0]).total_seconds()  # assumes uniform sampling
    #     shift_periods = int(round(delay_s / dt))

    #     # Shift labels backward so each row gets the label from `delay_s` seconds ahead
    #     df['binary_label_shifted'] = df[label_col_binary].shift(-shift_periods)
    #     df['discrete_label_shifted'] = df[label_col_discrete].shift(-shift_periods)
    #     df['continuous_label_shifted'] = df[label_col_continuous].shift(-shift_periods)

    #     # Rows at the end will have NaN labels — drop or handle as needed
    #     df = df.iloc[:-shift_periods] if shift_periods > 0 else df

    #     return df

        # df = aligned_df.copy()
        # delay = pd.Timedelta(seconds=delay_s)
        # df[f'binary_label_shifted'] = df[label_col_binary].shift(freq=delay)
        # df[f'discrete_label_shifted'] = df[label_col_discrete].shift(freq=delay)
        # df[f'continuous_label_shifted'] = df[label_col_continuous].shift(freq=delay)
        
        # return df

    def build_supervised_dataset(
        self, aligned_df: pd.DataFrame,
        fnirs_channels: List[str],
        label_col: str = 'label_shifted',
        window_duration_s: float = 6.0,
        resample_rate_hz: float = 10.0,
        use_shifted_data: bool = True,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct (X, y) for supervised learning from the aligned dataframe.

        A window of length `window_duration_s` is moved across the time axis with
        1-step stride, and the label at the window end is used as the target.
        """

        if use_shifted_data:
            binary_label_col = 'binary_'+label_col
            ternary_label_col = 'discrete_'+label_col
            continuous_label_col = 'continuous_'+label_col
        else:
            binary_label_col = 'binary_optimal'
            ternary_label_col = 'discrete_optimal'
            continuous_label_col = 'continuous_optimal'

        df = aligned_df.copy()
        df = df.dropna(subset=[binary_label_col, ternary_label_col, continuous_label_col])

        step_period_s = 1.0 / resample_rate_hz
        window_steps = int(round(window_duration_s / step_period_s))

        X_list: List[np.ndarray] = []
        y_list_binary, y_list_ternary, y_list_continuous = [], [], []

        values = df[fnirs_channels].to_numpy()
        labels_binary = df[binary_label_col].to_numpy()
        labels_ternary = df[ternary_label_col].to_numpy()
        labels_continuous = df[continuous_label_col].to_numpy()

        for end_idx in range(window_steps - 1, len(df)):
            start_idx = end_idx - window_steps + 1
            X_window = values[start_idx:end_idx + 1]
            feats = self.compute_window_features(X_window)
            X_list.append(feats)
            y_list_binary.append(labels_binary[end_idx])
            y_list_ternary.append(labels_ternary[end_idx])
            y_list_continuous.append(labels_continuous[end_idx])

        try:
            X = np.stack(X_list, axis=0)
        except:
            print(window_steps)
        y_binary = np.array(y_list_binary)
        y_ternary = np.array(y_list_ternary)
        y_continuous = np.array(y_list_continuous)

        

        return X, y_binary, y_ternary, y_continuous

    def compute_window_features(self, X: np.ndarray) -> np.ndarray:
        from scipy.stats import skew, kurtosis

        """compute_window_features(X: ndarray):
            Feature extraction for an offline window from the fnirs buffer [T, C] time x channels."""

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
        skewness = skew(X, axis=0, bias=False)
        kurt = kurtosis(X, axis=0, fisher=True, bias=False)

        # combine features
        features = np.concatenate([mean, std, slope, intercept, skewness, kurt], axis=0)
        return features