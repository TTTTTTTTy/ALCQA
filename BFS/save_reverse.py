import pickle
import json

graph = pickle.load(open('../data/bfs_data/wikidata.pkl','rb'))
rel2sub = {}
rel2obj = {}

for idx, g in enumerate(graph):
    qid = 'Q%d' % idx
    if 'sub' in g:
        for r in g['sub']:
            if r not in rel2sub:
                rel2sub[r] = []
            rel2sub[r].append(qid)
    if 'obj' in g:
        for r in g['obj']:
            if r not in rel2obj:
                rel2obj[r] = []
            rel2obj[r].append(qid)

json.dump(rel2sub, open('../data/bfs_data/rel2sub.json', 'w'))
json.dump(rel2obj, open('../data/bfs_data/rel2obj.json', 'w'))