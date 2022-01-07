import argparse
from itertools import repeat
import os
import shutil
from multiprocessing import Pool, Manager
import time
import zipfile

import numpy as np
import pandas as pd

from ml_microservice.conversion import Xml2Csv

def parse_timeseries(ts: str, tsDir: str, outputDir: str, result_dict):
    print(f"[*][{ts}] Composing \'{ts}\'")
    t0 = time.time()
    csv_dir = os.path.join(outputDir, f"s{ts[7:]}")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    xmls = os.listdir(tsDir)
    xmls.sort()
    xmls = [(xml, os.path.join(tsDir, xml)) for xml in xmls]
    #print(f"Xmls: {xmls}")
    for xml, xml_path in xmls:
        print(f"[*][{ts}:{xml}] - xml: \'{xml}\'")
        transf = Xml2Csv()
        with open(xml_path, "r") as f:
            content = "".join(f.read().replace("\n", ""))
        dfs = transf.parse(content)
        for dfid, df in dfs.items():
            csv_path = os.path.join(csv_dir, f"{dfid}.csv")
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False)
                continue
            print(f"[*][{ts}:{xml}]    - \'{dfid}\': Overriding older version")
            old_df = pd.read_csv(csv_path)
            old_df["Date"] = pd.to_datetime(old_df["Date"])
            cols = set()
            [cols.add(c) for c in old_df.columns]
            [cols.add(c) for c in df.columns]
            for c in cols:
                if c not in old_df:
                    old_df[c] = [np.nan]*len(old_df)
                if c not in df:
                    df[c] = [np.nan]*len(df)
            for _, row in df.iterrows():
                if pd.to_datetime("2021-04-26") <= row["Date"]:
                    break
                last_date = old_df.iloc[-1]["Date"]
                if (
                    last_date.year == row["Date"].year and 
                    last_date.month == row["Date"].month and
                    last_date.day == row["Date"].day
                ):
                    continue
                old_df = old_df.append(row, ignore_index=True)
            os.remove(csv_path)
            old_df.to_csv(csv_path, index=False)
    t1 = time.time()
    print("[*][{}] Done - took {:.2f}s".format(ts, t1-t0))
    result_dict[ts] = t1-t0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zipSeries", type=str)
    ap.add_argument("-od", type=str, default="timeseries")
    
    args = ap.parse_args()
    print("--- Compose timeseries --------")

    if os.path.exists(args.od):
        shutil.rmtree(args.od)
    os.makedirs(args.od)
    
    zip_dir = os.path.join(args.od, "zipContent")
    with zipfile.ZipFile(args.zipSeries, 'r') as zip_ref:
        zip_ref.extractall(zip_dir)
    ts_dirs = os.listdir(zip_dir) 
    print(f"[.] Series extracted: {ts_dirs}")

    totTime = pd.DataFrame()
    manager = Manager()
    return_dict = manager.dict()
    with Pool(processes=5) as pool:
        pool.starmap(parse_timeseries, zip(
            ts_dirs,
            [os.path.join(os.path.abspath(zip_dir), d) for d in ts_dirs],
            repeat(args.od),
            repeat(return_dict)
        ))
    pool.join()
    for ts, deltaT in return_dict.items():
        totTime = totTime.append({"series": ts, "time (s)": deltaT}, ignore_index=True)
    totTime.to_csv(os.path.join(args.od, "conversionTime.csv"), index=False)
    shutil.rmtree(zip_dir)