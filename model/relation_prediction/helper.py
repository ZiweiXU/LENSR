import torch


# bbox: [YMIN, YMAX, XMIN, XMAX]
def union_bbox(bbox1, bbox2):
    return [
        min(bbox1[0], bbox2[0]),
        max(bbox1[1], bbox2[1]),
        min(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3]),
    ]


def normalize_bbox(bbox, base_bbox):
    return [
        (bbox[0] - base_bbox[0]) / (base_bbox[1] - base_bbox[0]),
        (bbox[1] - base_bbox[0]) / (base_bbox[1] - base_bbox[0]),
        (bbox[2] - base_bbox[2]) / (base_bbox[3] - base_bbox[2]),
        (bbox[3] - base_bbox[2]) / (base_bbox[3] - base_bbox[2]),
    ]


def augment_bbox(x, info):
    x_augmented = torch.zeros([x.shape[0], x.shape[1] + 8]).cuda()
    for j in range(len(info)):
        bbox1 = info[j][1][1]
        bbox2 = info[j][2][1]
        bbox_u = union_bbox(bbox1, bbox2)
        bbox1_n = normalize_bbox(bbox1, bbox_u)
        bbox2_n = normalize_bbox(bbox2, bbox_u)
        x_augmented[j] = torch.cat((x[j], torch.cuda.FloatTensor(bbox1_n + bbox2_n)))
    return x_augmented