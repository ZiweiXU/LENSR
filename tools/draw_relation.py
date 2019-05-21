import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


# def draw_relation(file_name, annotation):
#
#     pass


def find_rel_id(anno, subject_id, object_id):
    rel_ids = []
    for idx, a in enumerate(anno):
        if a['object']['category'] == object_id and \
                a['subject']['category'] == subject_id:
            rel_ids.append(idx)
    return rel_ids

def draw_relation_bbox(file_name, annotation, predicates, objects, rel_id=None):
    # file_name = list(annotation.keys())[0]
    fig, ax = plt.subplots(1)
    img = mpimg.imread('../resource/VRD/sg_dataset/sg_dataset/sg_test_images/' + file_name)
    # Display the image
    ax.imshow(img)

    if type(rel_id) == int:
        rel_id = [rel_id]

    for i in range(0, len(annotation[file_name])):
        relation = predicates[annotation[file_name][i]['predicate']]
        object = objects[annotation[file_name][i]['object']['category']]
        object_coord = annotation[file_name][i]['object']['bbox']
        subject = objects[annotation[file_name][i]['subject']['category']]
        subject_coord = annotation[file_name][i]['subject']['bbox']

        # print(f'{i}, {relation}, {subject}, {object}')
        # print('-' * 25)

        if rel_id is not None and i not in rel_id:
            continue

        print(f'Ploted relationship: {subject}, {relation}, {object}')
        rect1 = patches.Rectangle(
            (object_coord[2], object_coord[0]),
            object_coord[3] - object_coord[2], object_coord[1] - object_coord[0],
            linewidth=5, edgecolor=[25/255,64/255,117/255,1], facecolor='none')
        rect2 = patches.Rectangle(
            (subject_coord[2], subject_coord[0]),
            subject_coord[3] - subject_coord[2], subject_coord[1] - subject_coord[0],
            linewidth=5, edgecolor=[227/255,135/255,66/255,1], facecolor='none')
        ax.add_patch(rect1)
        # ax.text(object_coord[2], object_coord[0], relation + f'({i})->' + object,
        #         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2},
        #         fontsize=8)
        ax.add_patch(rect2)
        # ax.text(subject_coord[2], subject_coord[0], subject + '->' + relation + f'({i})',
        #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 2},
        #         fontsize=8)
    plt.axis('off')
    plt.show()
    fig.savefig('plot.pdf', bbox='tight')


if __name__ == "__main__":
    annotation = json.load(open('../resource/VRD/annotations_test.json'))
    predicates = json.load(open('../resource/VRD/predicates.json'))
    objects = json.load(open('../resource/VRD/objects.json'))
    img_name = '6917705703_fb8151a7e7_b.jpg'
    rel_id = find_rel_id(annotation[img_name], 0, 97)
    # rel_id = None
    draw_relation_bbox(img_name, annotation, predicates, objects, rel_id=rel_id[1])
