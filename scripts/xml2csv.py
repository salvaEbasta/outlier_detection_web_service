import argparse
import glob
import logging
from multiprocessing import Pool
import os
import xml.etree.ElementTree as ElementTree
import re

import pandas as pd

ID = ['giorno_settimana', '.*_cd', '.*_CD', 'Abc_articolo']
DESCRIPTION = ['.*_ds', '.*_DS']


def _parse_row(row):
    desc = False
    rid, values = None, dict()
    for field in row:
        any_match = False
        if not desc:
            matches = [re.match(pattern, field.get('name'))
                       for pattern in DESCRIPTION]
            if any(matches):
                desc = True
                any_match = True
        if rid is None:
            matches = [re.match(pattern, field.get('name')) != None
                       for pattern in ID]
            if any(matches):
                rid = field.text
                any_match = True
        if not any_match:
            field_name = field.get('name')
            if len(field_name) == 0:
                field_name = 'empty'
            try:
                values[field_name] = float(field.text)
            except Exception as e:
                logging.warning(f"! - {rid}: {field_name}: {str(e)}")
                return None, dict()
    return rid, values


def convert_file(file, dframes=dict(), columns=set(), last_sample=None):
    logging.info(f"Last sample: {last_sample}")
    tree = ElementTree.parse(file)
    root = tree.getroot()
    for sample in root:
        logging.info(f"Sample id: {sample.get('id')}")
        if sample.get('id') == last_sample:
            logging.warning("! - Skip")
            continue
        last_sample = sample.get('id')
        rows = dict()
        for r in sample:
            rid, v = _parse_row(r)
            if rid != None:
                rows[rid] = v

        if len(rows) > 0 and len(dframes) == 0:
            for k in rows[next(iter(rows.keys()))].keys():
                logging.info(f"New csv: {k}")
                dframes[k] = dict()

        for rid, _ in rows.items():
            if rid not in columns:
                logging.info(f"New col: {rid}")
                columns.add(rid)
                for k, v in dframes.items():
                    if len(v.keys()) > 0:
                        v[rid] = [.0]*len(v[next(iter(v.keys()))])
                    else:
                        v[rid] = []

        not_updated = columns.copy()
        for rid, new_values in rows.items():
            for k, v in dframes.items():
                v[rid].append(new_values[k])
            not_updated.remove(rid)

        if len(not_updated) == len(columns):
            logging.warning("! - Pad ALL columns")
        for rid in not_updated:
            for _, v in dframes.items():
                v[rid].append(.0)
    logging.debug(f"{columns}, {dframes.keys()}")
    logging.debug(f"{[[len(v) for _, v in dframe.items()] for _, dframe in dframes.items()]}")
    return dframes, columns, last_sample


def convert_folder(root_dir):
    og_dir = os.getcwd()
    os.chdir(root_dir)
    xmls = glob.glob('*.xml')
    xmls.sort()

    dframes, columns, last_sample = dict(), set(), None

    for xml in xmls:
        xml = os.path.join(root_dir, xml)
        logging.info(f"[*] {xml}")
        dframes, columns, last_sample = convert_file(
            xml, dframes, columns, last_sample)

    os.chdir(og_dir)
    for k, v in dframes.items():
        pd.DataFrame(data=v).to_csv("{}.csv".format(k))


def convert_mono_folder(root_dir):
    og_dir = os.getcwd()
    os.chdir(root_dir)
    xmls = glob.glob('*.xml')
    xmls.sort()

    dframe, columns, last_sample = dict(), set(), None

    for xml in xmls:
        xml = os.path.join(root_dir, xml)
        logging.info(f"[*] {xml}")
        logging.info(f"Last sample: {last_sample}")
        tree = ElementTree.parse(xml)
        root = tree.getroot()
        for sample in root:
            logging.info(f"Sample id: {sample.get('id')}")
            if sample.get('id') == last_sample:
                logging.warning("! - Skip")
                continue
            last_sample = sample.get('id')
            rows = []
            for r in sample:
                _, v = _parse_row(r)
                if len(v) > 0:
                    rows.append(v)

            for r in rows:
                for k, v in r.items():
                    if k not in columns:
                        logging.info(f"New col: {k}")
                        columns.add(k)
                        if len(dframe.keys()) > 0:
                            dframe[k] = [.0]*len(dframe[next(iter(dframe.keys()))])
                        else:
                            dframe[k] = []

            not_updated = columns.copy()
            for r in rows:
                for k, v in r.items():
                    dframe[k].append(v)
                not_updated.remove(k)

            if len(not_updated) == len(columns):
                logging.info("! - Pad ALL columns")
            for vid in not_updated:
                dframe[vid].append(.0)
        logging.debug(f"{columns}, {dframe.keys()}")
        logging.debug(f"{[len(v) for _, v in dframe.items()]}")

    os.chdir(og_dir)
    pd.DataFrame(data=dframe).to_csv(
        "{}.csv".format(os.path.split(root_dir)[-1]))


def _explore_fields(root_dir):
    og_dir = os.getcwd()
    os.chdir(root_dir)
    xmls = glob.glob('*.xml')
    xmls.sort()
    # if(int(root_dir.split('_')[-1]) == 1):

    fields = set()
    for xml in xmls:
        #logging.info(f"[{root_dir}] Parsing - {xml}")
        tree = ElementTree.parse(xml)
        root = tree.getroot()
        for sample in root:
            for row in sample:
                for field in row:
                    if field.get('name') not in fields:
                        logging.info(f"New field: {field.get('name')}")
                        fields.add(field.get('name'))
    logging.debug(f"{root_dir}: fields: {fields}")


def bulk_explore(args):
    with Pool(args.pool_size) as p:
        p.map(_explore_fields, [
            os.path.join(args.dir, d)
            for d in os.listdir(args.dir)
            if os.path.isdir(os.path.join(args.dir, d))
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dir", type=str, help="Directory of .xml to parse",
                        default="C:\\Users\\Matteo\\Documents\\tesi\\series_9")
    parser.add_argument('-mono', action='store_true',
                        help="If set parse the xmls as they were a series with a single column")
    parser.add_argument('-ld', '--logger_dir', type=str, default='logs/xml2csv')
    args = parser.parse_args()

    # Logger setup
    if not os.path.exists(args.logger_dir):
        os.makedirs(args.logger_dir)
    log_f = os.path.join(args.logger_dir, os.path.split(args.dir)[-1])
    logging.basicConfig(format="%(levelname)s: %(message)s", 
                        filename="{}.log".format(log_f), 
                        encoding='utf-8', 
                        level=logging.INFO)

    if args.mono:
        logging.info("Mode: Monodimensional")
        convert_mono_folder(args.dir)
    else:
        convert_folder(args.dir)
