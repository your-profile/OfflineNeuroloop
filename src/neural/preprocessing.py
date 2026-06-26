import pandas as pd
import numpy as np
from typing import Tuple, List
import statistics
import scipy.stats

class DatasetProcessor:
    def __init__(self, verbose = False):
        self.neural_channels = None
        self.verbose = verbose
        self.gap_threshold_s = 600
        self.fnirs_df = None
        self.label_df = None

    def align_streams(self, 
                      fnirs_df: pd.DataFrame, 
                      task_df: pd.DataFrame, 
                      labels_df: pd.DataFrame, 
                      resample_rate_hz: float = 5.2, 
                      label_tolerance_s: float = 0.3, 
                      neural_channels: List[str] = ["L_O_DSI", "L_D_DSI", "L_O_DSphi", "L_D_DSphi", "R_O_DSI", "R_D_DSI", "R_O_DSphi", "R_D_DSphi"], 
                      gap_threshold_s: float = 600, 
                      verbose = False) -> Tuple[pd.DataFrame, List[str]]:

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

        aligned_segments = []
        # Align per participant first to avoid cross-participant time bleed.
        participant_col = None
        # Prefer numeric participant matching when available.
        if "pid" in fnirs_df.columns and "participant_id" in task_df.columns:
            participant_col = ("pid", "participant_id")
        elif "participant_id" in fnirs_df.columns and "participant_id" in task_df.columns:
            participant_col = ("participant_id", "participant_id")
        elif "pid" in fnirs_df.columns and "participantKey" in task_df.columns:
            participant_col = ("pid", "participantKey")

        if participant_col is None:
            fnirs_groups = [("__all__", fnirs_numeric)]
        else:
            fnirs_id_col, task_id_col = participant_col
            fnirs_groups = list(fnirs_numeric.groupby(fnirs_df[fnirs_id_col]))

        for participant_key, fnirs_part in fnirs_groups:
            if participant_col is None:
                task_part = task_df
                labels_part = labels_df
            else:
                _, task_id_col = participant_col
                task_part = task_df[task_df[task_id_col] == participant_key]
                # labels use 'pid' in this project
                if "pid" in labels_df.columns:
                    labels_part = labels_df[labels_df["pid"] == participant_key]
                elif "participant_id" in labels_df.columns:
                    labels_part = labels_df[labels_df["participant_id"] == participant_key]
                else:
                    labels_part = labels_df

            if fnirs_part.empty:
                continue

            gap = fnirs_part.index.to_series().diff().dt.total_seconds()
            segment_id = (gap > gap_threshold_s).cumsum()

            for seg in segment_id.unique():
                fnirs_seg = fnirs_part[segment_id == seg]
                if fnirs_seg.empty:
                    continue

                start = fnirs_seg.index.min()
                end = fnirs_seg.index.max()

                task_seg = task_part.loc[start:end]
                labels_seg = labels_part.loc[start:end]

                # remove duplicates
                if task_seg.index.has_duplicates:
                    task_seg = task_seg.groupby(level=0).last()
                if labels_seg.index.has_duplicates:
                    labels_seg = labels_seg.groupby(level=0).last()

                if task_seg.empty:
                    if self.verbose or verbose:
                        print(f"Skipping empty task segment for participant {participant_key}")
                    continue

                fnirs_resampled = fnirs_seg.resample(rule).mean().interpolate()
                task_resampled = task_seg.resample(rule).ffill()

                labels_resampled = labels_seg.reindex(
                    fnirs_resampled.index,
                    method='nearest',
                    tolerance=pd.Timedelta(seconds=label_tolerance_s)
                )
                labels_resampled = labels_resampled.ffill().infer_objects(copy=False)

                aligned = fnirs_resampled.join(task_resampled, how='left')
                aligned = aligned.join(labels_resampled, how='left', rsuffix='_label')
                aligned_segments.append(aligned)

                if self.verbose or verbose:
                    print(len(fnirs_resampled), len(task_resampled), len(labels_resampled))
          

        # concat
        if len(aligned_segments) == 0:
            aligned = pd.DataFrame(index=fnirs_numeric.index.copy())
        else:
            aligned = pd.concat([seg for seg in aligned_segments if not seg.empty]).sort_index()
        aligned.index.name = 'time'

        return aligned, fnirs_channels

    def compute_poisson_likelihood(self, granularity: str) -> pd.DataFrame:
        df = self.fnirs_df.copy()
        time_interval_s = (df.index[16] - df.index[0]).total_seconds()
        mu = 1.0 / time_interval_s
        events = df["binary_label_shifted"].values
        events = events[events != 0]
        likelihood_of_events = scipy.stats.poisson.pmf(events, mu)

        print("Likelihood of events: ", likelihood_of_events)
        print("Number of probabilities: ", len(likelihood_of_events))
        print("Time interval: ", time_interval_s)
        print("Number of original events: ", len(events))

        return likelihood_of_events


    def build_balanced_dataset(self, 
            aligned_df: pd.DataFrame,
            fnirs_channels: List[str],
            label_col: str = 'label_shifted',
            window_duration_s: float = 6.0,
            resample_rate_hz: float = 5.2,
            random_state: int | None = None,
            granularity: str = "binary",
            return_indices: bool = False,
            ):

        X, y_binary, y_ternary, y_continuous = self.build_supervised_dataset(
            aligned_df=aligned_df,
            fnirs_channels=fnirs_channels,
            label_col=label_col,
            window_duration_s=window_duration_s,
            resample_rate_hz=resample_rate_hz,
        )

        gs = str(granularity).strip().lower()
        mode = gs[0] if gs else "b"
        rng = np.random.default_rng(random_state)

        def balanced(class_indices, n_per: int):
            parts = [rng.choice(ix, size=n_per, replace=False) for ix in class_indices]
            sel = np.concatenate(parts)
            rng.shuffle(sel)
            return sel

        def pack(sel_arr: np.ndarray, y_out: np.ndarray):
            if return_indices:
                return X[sel_arr], y_out[sel_arr], sel_arr
            return X[sel_arr], y_out[sel_arr]

        if mode == "b":
            y = y_binary
            classes = np.unique(y)
            if len(classes) != 2:
                raise ValueError(f"Expected exactly 2 classes for binary label, got {classes}.")
            idx0 = np.flatnonzero(y == classes[0])
            idx1 = np.flatnonzero(y == classes[1])
            n_per = min(len(idx0), len(idx1))
            if n_per <= 0:
                raise ValueError("No samples in one or both binary classes.")
            sel = balanced([idx0, idx1], n_per)
            return pack(sel, y)

        if mode == "t":
            y = y_ternary
            classes, counts = np.unique(y, return_counts=True)
            if len(classes) != 3:
                raise ValueError(f"Expected exactly 3 classes for ternary label, got {classes}.")
            min_n = int(counts.min())
            med_n = statistics.median(counts.tolist())
            target_n = int(round((min_n + med_n) / 2))
            class_indices = [np.flatnonzero(y == c) for c in classes]
            n_per = min(target_n, *(len(ix) for ix in class_indices))
            if n_per <= 0:
                raise ValueError("No samples available for ternary balancing.")
            sel = balanced(class_indices, n_per)
            return pack(sel, y)

        if mode == "c":
            y = np.asarray(y_continuous, dtype=float)
            n = len(y)
            if n == 0:
                raise ValueError("No samples for continuous target.")
            edges = np.percentile(y, [100.0 / 3.0, 200.0 / 3.0])
            y_bin = np.digitize(y, edges, right=True)
            y_bin = np.clip(y_bin, 0, 2)
            class_indices = [np.flatnonzero(y_bin == k) for k in (0, 1, 2)]
            n_per = min(len(ix) for ix in class_indices)
            if n_per <= 0:
                sel = np.arange(n)
                rng.shuffle(sel)
                return pack(sel, y)
            sel = balanced(class_indices, n_per)
            return pack(sel, y)

        raise ValueError(
            f"Unknown granularity {granularity!r}; use 'binary', 'ternary', or 'continuous'."
        )

    def build_supervised_dataset_fnirs(self, 
                                 aligned_df: pd.DataFrame,
                                 fnirs_channels: List[str],
                                 window_duration_s: float = 6.0,
                                 resample_rate_hz: float = 10.0,
                                 step_size_s: int = 0) -> Tuple[np.ndarray, np.ndarray]:

        df = aligned_df.copy()
        df = df.dropna(subset=["states"])

        step_period_s = 1.0 / resample_rate_hz
        window_steps = int(round(window_duration_s / step_period_s))
        step_size = max(1, int(round(step_size_s / step_period_s)))

        fnirs_list: List[np.ndarray] = []
        s_list = []

        values = df[fnirs_channels].to_numpy()
        states = df["states"].to_numpy()

        for end_idx in range(window_steps - 1, len(df), step_size):
            start_idx = end_idx - window_steps + 1
            if start_idx < 0:
                continue
            fnirs_window = values[start_idx:end_idx + 1]
            feats = self.compute_window_features(fnirs_window)
            fnirs_list.append(feats)
            s_list.append(states[end_idx])

        try:
            F_list = np.stack(fnirs_list, axis=0)
        except:
            print(window_steps)

        S_list = np.array(s_list)

        return S_list, F_list

    def shift_labels_for_delay(self, 
                    aligned_df: pd.DataFrame,
                    delay_s: float,
                    label_col_binary: str = 'binary_optimal',
                    label_col_ternary: str = 'discrete_optimal',
                    label_col_continuous: str = 'continuous_optimal',
                    verbose: bool = False
            ) -> pd.DataFrame:

        df = aligned_df.copy()
        # normalize continuous labels to be between 0 and 1
        df[label_col_continuous] = (df[label_col_continuous] - df[label_col_continuous].min()) / (df[label_col_continuous].max() - df[label_col_continuous].min())
        
        dt = (df.index[1] - df.index[0]).total_seconds()
        shift_periods = int(round(delay_s / dt))

        # detect segment boundaries
        gaps = df.index.to_series().diff().dt.total_seconds()
        segment_id = (gaps > self.gap_threshold_s).cumsum()

        for col_shifted, col_orig in [('binary_label_shifted',   label_col_binary),
                                      ('ternary_label_shifted', label_col_ternary),
                                      ('continuous_label_shifted', label_col_continuous)]:
            df[col_shifted] = np.nan

            for seg in segment_id.unique():
                mask = segment_id == seg
                seg_labels = df.loc[mask, col_orig]
                df.loc[mask, col_shifted] = seg_labels.shift(-shift_periods).values

        self.label_df = df[['binary_label_shifted', 'ternary_label_shifted', 'continuous_label_shifted']].copy()
        df = df.dropna(subset=['binary_label_shifted', 'ternary_label_shifted', 'continuous_label_shifted'])

        self.fnirs_df = df

        return df

    @staticmethod
    def _iloc_nearest_sorted_ns(times_ns: np.ndarray, t_ns: np.int64) -> int:
        """Return iloc index of row whose index time is closest."""
        n = int(times_ns.shape[0])
        if n == 0:
            raise ValueError("empty dataframe for nearest time lookup")
        i = int(np.searchsorted(times_ns, t_ns, side="left"))
        if i <= 0:
            return 0
        if i >= n:
            return n - 1
        left = i - 1
        if (times_ns[i] - t_ns) < (t_ns - times_ns[left]):
            return i
        return left

    def _nearest_iloc_from_datetime_index(self, df: pd.DataFrame, cache_tag: str, target_time) -> int:
        df_id_attr = f"_tslookup_dfid_{cache_tag}"
        times_attr = f"_tslookup_times_ns_{cache_tag}"
        cur_id = id(df)
        if getattr(self, df_id_attr, None) != cur_id:
            setattr(self, df_id_attr, cur_id)
            setattr(self, times_attr, pd.DatetimeIndex(df.index).asi8.astype(np.int64, copy=False))
        times_ns = getattr(self, times_attr)
        t_ns = np.int64(pd.Timestamp(target_time).value)
        return self._iloc_nearest_sorted_ns(times_ns, t_ns)

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

        if "time" in self.fnirs_df.columns:
            closest_idx = int((self.fnirs_df["time"] - target_time).abs().argmin())
        else:
            closest_idx = self._nearest_iloc_from_datetime_index(self.fnirs_df, "fnirs", target_time)

        fnirs_neural_data = self.fnirs_df.iloc[closest_idx][fnirs_channels]

        return fnirs_neural_data

    def get_label_sample(self,
        timestamp,
        temporal_shift: float = 4.0,
        label_types: List[str] = ['binary_label_shifted', 'ternary_label_shifted', 'continuous_label_shifted']
    ):
        # Convert temporal_shift to Timedelta if it isn't already
        if not isinstance(temporal_shift, pd.Timedelta):
            temporal_shift = pd.Timedelta(seconds=temporal_shift)

        # Find closest fNIRS sample to (timestamp + temporal_shift)
        target_time = timestamp + temporal_shift

        if "time" in self.label_df.columns:
            closest_idx = int((self.label_df["time"] - target_time).abs().argmin())
        else:
            closest_idx = self._nearest_iloc_from_datetime_index(self.label_df, "label", target_time)

        label_data = self.label_df.iloc[closest_idx][label_types]

        if label_data.isna().any() or label_data.isnull().any():
            return label_data.fillna(0.0)

        return label_data

    def build_supervised_dataset(self, 
                                 aligned_df: pd.DataFrame,
                                 fnirs_channels: List[str],
                                 label_col: str = 'label_shifted',
                                 window_duration_s: float = 6.0,
                                 resample_rate_hz: float = 10.0,
                                 use_shifted_data: bool = True,
                                 step_size_s: int = 0) -> Tuple[np.ndarray, np.ndarray]:
      
        if use_shifted_data:
            binary_label_col = 'binary_'+label_col
            ternary_label_col = 'ternary_'+label_col
            continuous_label_col = 'continuous_'+label_col
        else:
            binary_label_col = 'binary_optimal'
            ternary_label_col = 'discrete_optimal'
            continuous_label_col = 'continuous_optimal'

        df = aligned_df.copy()
        df = df.dropna(subset=[binary_label_col, ternary_label_col, continuous_label_col])

        step_period_s = 1.0 / resample_rate_hz
        window_steps = int(round(window_duration_s / step_period_s))
        step_size = max(1, int(round(step_size_s / step_period_s)))

        X_list: List[np.ndarray] = []
        y_list_binary, y_list_ternary, y_list_continuous = [], [], []

        values = df[fnirs_channels].to_numpy()
        labels_binary = df[binary_label_col].to_numpy()
        labels_ternary = df[ternary_label_col].to_numpy()
        labels_continuous = df[continuous_label_col].to_numpy()

        # Sliding window with configurable step_size
        for end_idx in range(window_steps - 1, len(df), step_size):
            start_idx = end_idx - window_steps + 1
            if start_idx < 0:
                continue
            X_window = values[start_idx:end_idx + 1]
            feats = self.compute_window_features(X_window)
            X_list.append(feats)
            y_list_binary.append(labels_binary[end_idx])
            y_list_ternary.append(labels_ternary[end_idx])
            y_list_continuous.append(labels_continuous[end_idx])

        try:
            X = np.stack(X_list, axis=0)
        except Exception as e:
            print(f"Error stacking X_list, window_steps={window_steps}, error: {e}")

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