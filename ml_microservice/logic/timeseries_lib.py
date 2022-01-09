import logging
import os
import re

import numpy as np
import pandas as pd

from ml_microservice import configuration as cfg

class TimeseriesLibrary:
    def __init__(self, 
        path = cfg.timeseries.path, 
        date_col = cfg.timeseries.date_column,
        value_col = cfg.timeseries.value_column,
    ):
        self.logger = logging.getLogger('tsLib')
        self.logger.setLevel(logging.DEBUG)
        self.storage = path
        self.date_col = date_col
        self.value_col = value_col

    @property
    def timeseries(self):
        """
        -> [{group: str, dimensions: [str, ...]}, ...]
        """
        groups = []
        for f in os.listdir(self.storage):
            abs_f = os.path.join(self.storage, f)
            if os.path.isdir(abs_f):
                groups.append({
                    "group": f,
                    "dimensions": [
                        csv[:-4] for csv in os.listdir(abs_f)
                            if re.match('.*.csv', csv) is not None
                    ],
                })
        return groups

    def _2storage_path(self, group: str, dim: str = None):
        """-> (path: str, exists: bool)"""
        stg_path = os.path.join(self.storage, group)
        if dim is None:
            return stg_path, os.path.exists(stg_path)
        stg_path = os.path.join(stg_path, "{:s}.csv".format(dim))
        return stg_path, os.path.exists(stg_path)

    def fetch(self, group :str, dim: str):
        """ -> pandas.DataFrame """
        self.logger.info("Fetch {}/{}".format(group, dim))
        if not self.has(group, dim):
            return None
        df = pd.read_csv(self._2storage_path(group, dim)[0])
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df

    def fetch_ts(self, group: str, dim: str, tsID: str):
        """-> pd.Dataframe"""
        self.logger.info("[.] Fetch ts: {:s}/{:s}-{:s}".format(group, dim, tsID))
        if not self.has(group, dim):
            self.logger.warning("[!] {Group: \'{:s}\', Dim: \'{:s}\'} not found".format(
                group,
                dim
            ))
            return None
        df = self.fetch(group, dim)
        if tsID not in df.columns:
            self.logger.warning("[!] tsID {:s} not in {:s}/{:s}".format(tsID, group, dim))
            return None
        ts = pd.DataFrame()
        ts[self.date_col] = df[self.date_col]
        ts[self.value_col] = df[tsID]
        return ts

    def has_group(self, group: str):
        return self._2storage_path(group)[-1]

    def has_dimension(self, dim: str):
        for j in self.timeseries:
            for d in j["dimensions"]:
                if dim == d:
                    return True
        return False

    def has(self, group:str, dimension: str):
        return self._2storage_path(group, dimension)[-1]

    def remove(self, group: str, dimension: str, tsID: str = None):
        df_path, is_real = self._2storage_path(group)
        if not is_real:
            return True
        df_path, is_real = self._2storage_path(group, dimension)
        if not is_real:
            return True
        if tsID is None:
            os.remove(df_path)
            return True
        if tsID == self.date_col:
            logging.warning("[!] Asked to remove date ts, refused")
            return False
        df = pd.read_csv(df_path)
        if tsID not in df.columns:
            return True
        df = df.drop(tsID, axis=1)
        os.remove(df_path)
        df.to_csv(df_path, index=False)
        return True

    def save(self, group: str, dfID: str, df: pd.DataFrame, override: bool = False):
        """ No merging """
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        if not override and self.has(group, dfID):
            old_df = self.fetch(group, dfID)
            cols = set()
            [cols.add(c) for c in old_df.columns]
            [cols.add(c) for c in df.columns]
            for c in cols:
                if c not in old_df:
                    old_df[c] = [np.nan]*len(old_df)
                if c not in df:
                    df[c] = [np.nan]*len(df)
            for _, row in df.iterrows():
                last_date = old_df.iloc[len(old_df) - 1][self.date_col]
                if (
                    last_date.year == row[self.date_col].year and 
                    last_date.month == row[self.date_col].month and
                    last_date.day == row[self.date_col].day
                ):
                    continue
                old_df.loc[len(old_df)] = row
                #old_df = old_df.append(row, ignore_index=True)
            df = old_df
        self.remove(group, dfID)
        df_path = self._2storage_path(group, dfID)[0]
        df.to_csv(df_path, index=False)
        return True
