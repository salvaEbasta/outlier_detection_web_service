import logging
import re
import time
import xml.etree.ElementTree as ElementTree

import numpy as np
import pandas as pd

from ml_microservice import configuration

IGNORES = configuration.xml.ignore
IDS = configuration.xml.ids
UNNAMED = configuration.xml.empty_field_name
DATE_COL = configuration.timeseries.date_column

logger = logging.getLogger("xml2csv")
logger.setLevel(logging.INFO)

class RowAggregator():
    def __init__(self, date_col = DATE_COL):
        self.date_col = date_col
        self.dfs = {}
        self.dates = []

    def dump(self, date: str, sample):
        """date, sample: [(rid, values: dict), ]"""
        #date = pd.to_datetime(date, self.date_format)
        date = pd.to_datetime(date)
        updates = {}
        for rid, values in sample:
            for dim, v in values.items():
                if dim not in updates.keys():
                    updates[dim] = {}
                if dim not in self.dfs.keys():
                    logging.info(f"[*] - new dataframe: {dim}")
                    self.dfs[dim] = pd.DataFrame()
                    self.dfs[dim][self.date_col] = list(self.dates)
                if rid is None:
                    rid = dim
                updates[dim][rid] = v
        logging.debug(f"Updates: {updates}")

        for dim in self.dfs.keys():
            df = self.dfs[dim]
            update = updates[dim] if dim in updates else {}
            cols = set()
            for c in update.keys():
                cols.add(c)
            for c in df.columns:
                cols.add(c)
            update[self.date_col] = date
            for c in cols:
                if c not in update.keys():
                    update[c] = np.nan
                if c not in df.columns:
                    logging.info(f"[*] - dataframe \'{dim}\': new feature \'{c}\'")
                    df[c] = [np.nan]*len(self.dates)
            self.dfs[dim] = df.append(update, ignore_index=True)
        self.dates.append(date)
        logging.debug(f"Dates: {self.dates}")
        logging.debug(f"Dfs: {self.dfs}")

    @property
    def result(self):
        return self.dfs

class Xml2Csv():
    def __init__(self, 
        default_unnamed = UNNAMED, 
        ignore_list: list = IGNORES, 
        id_patterns: list = IDS,
    ):
        self._ignore_list = ignore_list if len(ignore_list) else IGNORES
        self._ids = id_patterns if len(ignore_list) else IDS
        logging.debug(f"ids: {self._ids}, ignores: {self._ignore_list}")

        self._default_unamed = default_unnamed
        self._t_start = 0
        self._t_end = 0
    
    def _parse_row(self, row):
        rid = None
        values = {}
        for field in row:
            if any([re.match(pattern, field.get("name")) != None
                        for pattern in self._ignore_list]):
                continue
            if rid is None and any([re.match(pattern, field.get("name")) != None
                        for pattern in self._ids]):
                rid = field.text
                continue
            fname = field.get("name") 
            if len(fname) == 0:
                fname = self._default_unamed
            if fname in values.keys():
                raise ValueError("XML format: row has 2+ fields with same name")
            if field.text == "None":
                values[fname] = np.nan
                continue
            values[fname] = float(field.text)
        return rid, values

    def parse(self, xml: str):
        """ -> dict: {id: pd.Dataframe, } """
        logging.info(f'[.] xml2csv: id: {self._ids}, ignore: {self._ignore_list}')
        self._t_start = time.time()
        self._t_end = time.time()

        rowAgg = RowAggregator()
        #rowAgg.set_mode(self.is_mono(xml))
        
        data = ElementTree.fromstring(xml)
        last_date = None
        for sample in data:
            logging.info(f"[*] Sample: {sample.get('id')}")
            if sample.get('id') == last_date:
                logging.info(f"[!] Duplicate : {sample.get('id')}: skip")
                continue
            last_date = sample.get('id')

            rows = []
            for row in sample:
                rid, values = self._parse_row(row)
                rows.append((rid, values))
            rowAgg.dump(last_date, rows)
        
        self._t_end = time.time()
        return rowAgg.result
    
    @property
    def duration(self):
        return self._t_end - self._t_start

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-od", type=str, default=".")
    ap.add_argument("xmlPath", type=str)
    args = ap.parse_args()
    with open(args.xmlPath, "r") as f:
        content = "".join(f.read().replace("\n", ""))

    transf = Xml2Csv()
    dfs = transf.parse(content)

    import os
    if not os.path.exists(args.od):
        os.makedirs(args.od)
    for dfid, df in dfs.items():
        csv_path = os.path.join(args.od, f"{dfid}.csv")
        df.to_csv(csv_path, index=False)
