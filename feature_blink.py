import pandas as pd
import pymongo
import multiprocessing
import tqdm
import sys
import os
import concurrent.futures
import json
import string
from nltk.corpus import stopwords

import blink.main_dense as main_dense
import argparse
import scipy
import os

manager = multiprocessing.Manager()

logname = os.path.basename(__file__) + '.log'

if os.path.exists(logname):
  os.remove(logname)
# # 建立mongodb连接
client = pymongo.MongoClient(host='localhost', port=27017)

db = client.dbpedia

features = db.features
# # 连接stock数据库，注意只有往数据库中插入了数据，数据库才会自动创建
config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 100,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}


datasets = ['train', 'testA', 'testB']
datasets = [os.getenv("HOME") + f'/lnn-el/data/aida/template/full_{name}.csv' for name in datasets]


