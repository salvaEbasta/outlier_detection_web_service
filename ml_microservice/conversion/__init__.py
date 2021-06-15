import logging
import os
import re
from typing import List, Dict
from xml.etree import ElementTree as ET

import pandas as pd

from ml_microservice import constants

logger = logging.getLogger('xml2csv')
logger.setLevel(logging.INFO)

class Dataset():
    def __init__(self, values: Dict[str, List] = None):
        self.__init_dframe(values)
    
    def __init_dframe(self, values):
        self.dframe = dict()
        self._length = 0
        if values is not None:
            for k, v in values:
                self.dframe[k] = v
    
    @property
    def values(self):
        return self.dframe

    @property
    def length(self):
        return self._length

    @property
    def columns(self):
        return list(self.dframes.keys())
    
    def put(self, col, value):
        if col not in self.dframe.keys():
            logger.info(f"[.] - New column: {col}")
            self.dframe[col] = [.0]*self.length
        self.dframe[col].append(value)

    def pad(self, increment=1):
        self._length += increment
        for k in self.dframe.keys():
            self.dframe[k].extend([.0]*(self._length - len(self.dframe[k])))

class Aggregator():
    def __init__(self, mono_dset_name: str = constants.xml.empty_field_name):
        self._mono_dset_name = mono_dset_name
        self._dsets = dict()
        self.set_mode()

    def _dump_mono(self, rid, dim, value):
        if not self._mono_dset_name in self._dsets:
            logger.info("[.] - New dataset: {:s}".format(self._mono_dset_name))
            self._dsets[self._mono_dset_name] = Dataset()
        self._dsets[self._mono_dset_name].put(dim, value)
        
    def _dump_std(self, rid, dim, value):
        if not dim in self._dsets:
            logger.info("[.] - New dataset: {:s}".format(dim))
            length = 0
            if len(self._dsets) > 0:
                length = self._dsets[next(iter(self._dsets.keys()))].length
            self._dsets[dim] = Dataset()
            self._dsets[dim].pad(increment=length)
        self._dsets[dim].put(rid, value)

    def set_mode(self, mono = False):
        self._mono = mono
        if mono:
            self._dump = self._dump_mono
        else:
            self._dump = self._dump_std
    
    def dump(self, rows):
        for updates in rows:
            for rid, dim, value in updates:
                self._dump(rid, dim, value)
        for dset in self._dsets.values():
            dset.pad()

    @property
    def result(self):
        return [(k, v.values) for k, v in self._dsets.items()]

class Xml2Csv():
    def __init__(self, ignore_list = constants.xml.ignore, 
                        id_list = constants.xml.ids, default_unnamed = constants.xml.empty_field_name):
        self._ignore = ignore_list
        self._id = id_list
        logger.debug("default ids: {}, default ignore: {}".format(self._id, self._ignore))
        
        self._default_empty = default_unnamed

    def _parse_row(self, row):
        # updates: [(rid, dim, value), ...]
        updates = list()
        rid = None
        for field in row:
            any_match = False
            matches_ignore = [re.match(pattern, field.get('name'))
                                for pattern in self._ignore]
            if any(matches_ignore):
                any_match = True
            if rid is None:
                matches = [re.match(pattern, field.get('name')) != None
                           for pattern in self._id]
                if any(matches):
                    rid = field.text
                    updates = [(rid, t[1], t[2]) for t in updates]
                    any_match = True
            if not any_match:
                field_name = field.get('name')
                if len(field_name) == 0:
                    if self._default_empty in [t[1] for t in updates]:
                        raise ValueError('The xml contains 2+ fields with attr .name left empty')
                    field_name = self._default_empty
                try:
                    updates.append((rid, field_name ,float(field.text)))
                except Exception as e:
                    logger.warning(f"[!] {rid}: {field_name}: {str(e)}")
                    return []
        return updates

    def is_mono(self, xml: str):
        root = ET.fromstring(xml)
        id_found = None
        for sample in root:
            for row in sample:
                id_found = False
                for field in row:
                    id_pattern_matches = [re.match(pattern, field.get('name')) != None
                                            for pattern in self._id]
                    if any(id_pattern_matches):
                        id_found = True
                        break
                break
            if id_found is not None:
                break
        logging.info(f"[.] Mono dim: {not id_found}")
        return not id_found

    def parse(self, xml: str, id_field: str = None, to_ignore: List[str] = None):
        """ -> [(name, dict), ...] """
        logger.info('[.] xml2csv: id: {}, ignore: {}'.format(id_field, to_ignore))
        if id_field is not None:
            self._id = [id_field]
        logger.info('[.] ID field.name(s): {}'.format(self._id))

        if to_ignore is not None:
            self._ignore = to_ignore
        logger.info('[.] Ignore field.name(s): {}'.format(self._ignore))
        
        agg = Aggregator()
        agg.set_mode(self.is_mono(xml))
        
        root = ET.fromstring(xml)
        last_sample = None
        for sample in root:
            if sample.get('id') == last_sample:
                logger.warning(f"[!] Duplicate : {sample.get('id')}: skip")
                continue
            logger.info(f"[*] Sample: {sample.get('id')}")
            last_sample = sample.get('id')

            rows = []
            for row in sample:
                rows.append(self._parse_row(row))
            agg.dump(rows)
        return agg.result
