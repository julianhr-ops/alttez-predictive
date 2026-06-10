import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from alttez.features.selection import snap_split_index
from alttez.features.target import construct_targets
from alttez.models.model import ClassificationModel


class Trainer:
    """ Classification Model trainer class. """

    @staticmethod
    def _resolve_date_column(df: pd.DataFrame) -> str:
        for candidate in ("date", "Date"):
            if candidate in df.columns:
                return candidate
        raise ValueError('Expected dataframe to contain a "date" or "Date" column for temporal validation.')

    def train(
            self,
            model: ClassificationModel,
            train_df: pd.DataFrame,
            eval_df: Optional[pd.DataFrame] = None,
            check_nan: bool = True
    ) -> Tuple[ClassificationModel, pd.DataFrame]:
        """ Fits the model in the provided dataset. """

        # Clean Dataframe.
        eval_has_nan = eval_df.isna().any().any() if eval_df is not None else False
        if check_nan and (train_df.isna().any().any() or eval_has_nan):
            raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

        metrics_df = model.fit(train_df=train_df, eval_df=eval_df)
        return model, metrics_df

    def cross_validation(
            self,
            model: ClassificationModel,
            df: pd.DataFrame,
            k_folds: int = 5
    ) -> pd.DataFrame:
        """ Evaluates the model using the Stratify k-fold cross validation method. """

        # Clean Dataframe.
        if df.isna().any().any():
            raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

        targets = construct_targets(df=df, target_type=model.target_type)

        # Initialize stratified k-fold procedure.
        metrics_df_per_fold = []
        cv_generator = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0).split(df, targets)
        for i, (train_ids, eval_ids) in tqdm(iterable=enumerate(cv_generator), desc='Evaluating K-Fold', total=k_folds):
            train_df = df.iloc[train_ids]
            eval_df = df.iloc[eval_ids]
            model, metrics_df = self.train(model=model, train_df=train_df, eval_df=eval_df, check_nan=False)
            metrics_df['Fold'] = i + 1
            metrics_df_per_fold.append(metrics_df)
        cv_df = pd.concat(metrics_df_per_fold, ignore_index=True, axis=0)
        return cv_df

    def sliding_cross_validation(
            self,
            model: ClassificationModel,
            df: pd.DataFrame,
            test_ratio: float,
            k_folds: int = 5
    ) -> pd.DataFrame:
        """ Backwards-compatible alias for temporal expanding-window cross validation. """

        return self.temporal_cross_validation(model=model, df=df, test_ratio=test_ratio, k_folds=k_folds)

    def temporal_cross_validation(
            self,
            model: ClassificationModel,
            df: pd.DataFrame,
            test_ratio: float = 0.2,
            k_folds: int = 5
    ) -> pd.DataFrame:
        """ Evaluates the model using expanding-window temporal cross validation. """

        if df.isna().any().any():
            raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

        date_column = self._resolve_date_column(df)
        ordered_df = df.sort_values(date_column, ascending=True).reset_index(drop=True)
        total_samples = ordered_df.shape[0]
        if total_samples < 3:
            raise ValueError('Temporal cross validation requires at least 3 samples.')

        if k_folds < 1:
            raise ValueError('k_folds must be greater than zero.')

        estimated_eval_size = max(1, int(round(total_samples * float(test_ratio))))
        max_possible_folds = max(1, (total_samples - 1) // estimated_eval_size)
        folds = min(k_folds, max_possible_folds)
        eval_size = max(1, min(estimated_eval_size, (total_samples - 1) // folds))
        initial_train_size = total_samples - (folds * eval_size)
        if initial_train_size <= 0:
            initial_train_size = 1
            eval_size = max(1, (total_samples - initial_train_size) // folds)

        metrics_df_per_fold = []
        for fold_index in tqdm(iterable=range(folds), desc='Evaluating Temporal K-Fold', total=folds):
            raw_train_end = initial_train_size + (fold_index * eval_size)
            raw_eval_end = min(raw_train_end + eval_size, total_samples)

            # Snap to date boundaries so a single date never straddles the split.
            train_end = snap_split_index(ordered_df, raw_train_end, date_column)
            eval_start = train_end
            eval_end = snap_split_index(ordered_df, raw_eval_end, date_column)
            if train_end <= 0 or eval_start >= total_samples or eval_end <= eval_start:
                continue

            train_df = ordered_df.iloc[:train_end].reset_index(drop=True)
            eval_df = ordered_df.iloc[eval_start:eval_end].reset_index(drop=True)
            model, metrics_df = self.train(model=model, train_df=train_df, eval_df=eval_df, check_nan=False)
            metrics_df['Fold'] = fold_index + 1
            metrics_df['Start Date'] = [train_df.iloc[0][date_column], eval_df.iloc[0][date_column]]
            metrics_df['End Date'] = [train_df.iloc[-1][date_column], eval_df.iloc[-1][date_column]]
            metrics_df['Samples'] = [train_df.shape[0], eval_df.shape[0]]
            metrics_df_per_fold.append(metrics_df)

        if not metrics_df_per_fold:
            raise ValueError('Temporal cross validation could not produce any folds.')

        cv_df = pd.concat(metrics_df_per_fold, ignore_index=True, axis=0)
        return cv_df
