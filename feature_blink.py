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
models_path = "models/" # the path where you stored the BLINK models

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

args = argparse.Namespace(**config)
models = main_dense.load_models(args, logger=None)

def blink_process(set_to_calculate):
	# doc, men, left, right, candidates
	global feature_list
	try:
		data_to_link = [
			{
				"id": i,
				"label": "unknown",
				"label_id": -1,
				"context_left": entry[2].lower(),
				"mention": entry[1].lower(),
				"context_right": entry[3].lower(),

			}
			for i, entry in enumerate(set_to_calculate)
		]
		print(len(data_to_link))
		_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
		# scores = scipy.special.softmax(scores)
		# predictions = {i: {pred: scores[i][j] for j, pred in enumerate(prediction)} for i, prediction in enumerate(predictions)}
		# for i, entry in 
		# print(predictions)

		
	except KeyboardInterrupt:
		raise KeyboardInterrupt
	except:
		import traceback
		traceback.print_exc()
		with open(logname, 'a') as f:
			f.write(f'{set_to_calculate}\n')

def fetch_candidate(mentionLabel):
	_, cand = mentionLabel.split('===')
	return cand

datasets = ['train', 'testA', 'testB']
datasets = [os.getenv("HOME") + f'/lnn-el/data/aida/template/full_{name}.csv' for name in datasets]


known_feature_set = set()
feature_list = list(features.find({'t_': 'b1'}))

set_to_calculate = []
for entry in tqdm.tqdm(feature_list):
	known_feature_set.add((entry['d'],entry['men'], entry['left']))
del feature_list
feature_dict = manager.list()

for dataset in datasets:
	df_ = pd.read_csv(dataset).head(200)
	n, l, r = df_.shape[0], 0, 0
	pbar = tqdm.tqdm(total = len(set(df_.QuestionMention.values)))
	while r < n:
		while r < n - 1 and df_.iloc[l]['QuestionMention'] == df_.iloc[r + 1]['QuestionMention']:
			r+= 1
		batch = df_.iloc[l : r + 1]
		first_line = batch.iloc[0]
		# print(first_line)
		doc, left = first_line.QuestionMention.split('===')
		right = first_line.right
		men = first_line.Mention

		
		l = r + 1
		r = l
		pbar.update(1)
		if (doc, men, left) in known_feature_set: continue
		candidates = batch.Mention_label.apply(fetch_candidate)
		# print(batch.shape[0])
		set_to_calculate.append((doc, men, left,right, candidates))
		# print(candidates.Mention_label.values)
		# print(doc, left)
		# gold_pairs = batch[batch.Label.eq(1)]['Mention_label'].values
		# assert(len(gold_pairs) == 1)
		# gt = gold_pairs[0].split(';')[1].replace(' ', '_')
	pbar.close()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def save():
	pass

set_to_calculate = chunks(set_to_calculate, 4)


print(f'Process chunks of size {len(set_to_calculate)}')
try:
	with multiprocessing.Pool(40) as pool:
		[ _ for _ in tqdm.tqdm(pool.imap_unordered(blink_process, set_to_calculate), total = len(set_to_calculate))]
except KeyboardInterrupt:
	save()
	sys.exit()
save()



