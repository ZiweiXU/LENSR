import numpy as np
import json
from model.Misc.Formula import *

with open('./dataset_info/objects.json') as f:
    objects = json.load(f)
with open('./dataset_info/predicates.json') as f:
    predicates = json.load(f)
with open('./dataset_info/annotations_train.json') as f:
    annotations = json.load(f)
print (len(predicates),len(objects))

all_relations = set()
all_relations_bbox = {}
for k, v in annotations.items():
    for i in range(len(v)):
        all_relations.add((v[i]['predicate'],v[i]['subject']['category'],v[i]['object']['category']))
        # all_relations_bbox.add((v[i]['predicate'], v[i]['subject']['category'], v[i]['object']['category'],box_prop(v[i]['subject']['bbox'],v[i]['object']['bbox'])))

        key = (v[i]['predicate'], box_prop(v[i]['subject']['bbox'], v[i]['object']['bbox']))
        if key not in all_relations_bbox:
            all_relations_bbox[key] = [k]
        else:
            all_relations_bbox[key].append(k)

# print(len(all_relations),len(all_relations_bbox))

box_names = ['out','in','o_left','o_right','o_up','o_down','n_left','n_right','n_up','n_down']

sorted_relations = sorted(list(all_relations_bbox.keys()),key=lambda x: x[0])

english = []
for i in range(len(sorted_relations)):
    english.append([predicates[sorted_relations[i][0]],sorted_relations[i][0],box_names[sorted_relations[i][1]],sorted_relations[i][1],len(all_relations_bbox[sorted_relations[i]])])
print (english)

print (all_relations_bbox[(0,6)])

print (annotations['9728584093_110c2bea9d_b.jpg'])