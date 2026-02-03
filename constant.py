DEFAULT_EPS_LIST = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.17, 0.20]

COCO_TO_VOC = {
    4: 0,  # airplane    -> aeroplane
    1: 1,  # bicycle     -> bicycle
    14: 2,  # bird        -> bird
    8: 3,  # boat        -> boat
    39: 4,  # bottle      -> bottle
    5: 5,  # bus         -> bus
    2: 6,  # car         -> car
    15: 7,  # cat         -> cat
    56: 8,  # chair       -> chair
    19: 9,  # cow         -> cow
    60: 10,  # dining table -> diningtable
    16: 11,  # dog         -> dog
    17: 12,  # horse       -> horse
    3: 13,  # motorcycle  -> motorbike
    0: 14,  # person      -> person
    58: 15,  # potted plant -> pottedplant
    18: 16,  # sheep       -> sheep
    57: 17,  # couch       -> sofa
    6: 18,  # train       -> train
    62: 19,  # tv          -> tvmonitor
}

VOC_CLASS_COUNT = 20

# VOC classes
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

INPUT_SIZE = 640

COCO_TO_VOC_SSD = {
    5: 0,  # airplane    -> aeroplane
    2: 1,  # bicycle     -> bicycle
    16: 2,  # bird        -> bird
    9: 3,  # boat        -> boat
    44: 4,  # bottle      -> bottle
    6: 5,  # bus         -> bus
    3: 6,  # car         -> car
    17: 7,  # cat         -> cat
    62: 8,  # chair       -> chair
    21: 9,  # cow         -> cow
    67: 10,  # dining table -> diningtable
    18: 11,  # dog         -> dog
    19: 12,  # horse       -> horse
    4: 13,  # motorcycle  -> motorbike
    1: 14,  # person      -> person
    64: 15,  # potted plant -> pottedplant
    20: 16,  # sheep       -> sheep
    63: 17,  # couch       -> sofa
    7: 18,  # train       -> train
    72: 19,  # tv          -> tvmonitor
}
