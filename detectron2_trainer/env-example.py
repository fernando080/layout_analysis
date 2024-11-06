# Paths for dataset and processed files
annotations_path = "./dataset/dataset_name/result.json"
new_image_path = "/path/to/images"
new_file_path = "./dataset/dataset_name/filtered_result.json"
augment_file = "./dataset/dataset_name/filtered_result_train.json"

path_detectron2_train_output = "./path/to/detectron2/train/output"
path_layoutlmv3_train_output = "./path/to/layoutlmv3/train/output"

# Name of the subset to be used
subset_name = "dataset_subset_name"

# Categories for labeling
categories = {
    "0": "CATEGORY_A",
    "1": "CATEGORY_B",
    "2": "CATEGORY_C",
    "3": "CATEGORY_D",
    "4": "CATEGORY_E",
    "5": "CATEGORY_F",
    "6": "CATEGORY_G",
    "7": "CATEGORY_H",
    "8": "CATEGORY_I",
    "9": "CATEGORY_J",
    "10": "CATEGORY_K",
    "11": "CATEGORY_L",
    "12": "CATEGORY_M",
    "13": "CATEGORY_N",
    "14": "CATEGORY_O",
    "15": "CATEGORY_P",
    "16": "CATEGORY_Q",
    "17": "CATEGORY_R",
    "18": "CATEGORY_S",
    "19": "CATEGORY_T"
}

# Categories to filter
categories_to_filter = {
    "1": "CATEGORY_B",
    "2": "CATEGORY_C",
    "6": "CATEGORY_G",
    "13": "CATEGORY_N",
    "14": "CATEGORY_O"
}

# Categories to use for training
categories_to_train = {
    "1": "CATEGORY_B",
    "2": "CATEGORY_C",
    "6": "CATEGORY_G",
    "13": "CATEGORY_N",
    "14": "CATEGORY_O",
    "15": "CATEGORY_P",
    "16": "CATEGORY_Q"
}

# Category ID transformations
categories_transformations = {
    "1": "0",
    "2": "1",
    "6": "2",
    "13": "3",
    "14": "4",
    "15": "5",
    "16": "6"
}

# If you use the library augraphy, you can use the following dictionary to map the augmentations to the categories
augmentations_archetype = {
    "1": "_aug_archetype_01",
    "2": "_aug_archetype_02",
    "3": "_aug_archetype_03",
    "4": "_aug_archetype_04",
    "5": "_aug_archetype_05",
    "6": "_aug_archetype_06",
    "7": "_aug_archetype_07",
    "9": "_aug_archetype_009",
    "10": "_aug_archetype_010",
    "11": "_aug_archetype_011"
}