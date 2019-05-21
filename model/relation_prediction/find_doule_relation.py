import json

with open('../dataset_info/annotations_test.json', 'r') as f:
    annotation_test = json.load(f)

for key in annotation_test:
    pairs = {}
    for annotation in annotation_test[key]:
        if (annotation['subject']['category'],annotation['object']['category']) not in pairs.keys():
            pairs[(annotation['subject']['category'],annotation['object']['category'])] = 0
            pairs[(annotation['object']['category'], annotation['subject']['category'])] = 0
        else:
            pairs[(annotation['subject']['category'], annotation['object']['category'])] += 1
            print (key,(annotation['subject']['category'], annotation['object']['category']))
