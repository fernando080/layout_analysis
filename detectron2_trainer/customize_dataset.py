"""
This script stores functions to personalize the annotations of the dataset, to use the labels you considere more
appropriate, you can use the function filter_and_prepare_labels.
"""
import os
import env
import json

import tqdm


def filter_categories_and_update_filenames(file_path,
                                           categories_to_filter,
                                           categories_to_train,
                                           new_image_path,
                                           new_file_path,
                                           use_augmentation=False,
                                           ):
    # Read the JSON file
    print("Reading JSON file...")
    with open(file_path, 'r') as file:
        coco_dict = json.load(file)

    # Filter categories
    print("Filtering categories...")
    filtered_categories = [category for category in coco_dict["categories"] if
                           str(category["id"]) in categories_to_train]
    # Use dict in env.categories_transformations to transform the categories ids
    # filtered_categories_transformed =
    for cat in filtered_categories:
        cat["id"] = int(env.categories_transformations[str(cat["id"])])

    coco_dict["categories"] = filtered_categories

    # Create a set of category IDs to filter
    print("Creating category IDs set...")
    category_ids = set(categories_to_train.keys())

    # Filter annotations based on the allowed categories
    print("Filtering annotations...")
    filtered_annotations = [annotation for annotation in coco_dict["annotations"] if
                            str(annotation["category_id"]) in category_ids]
    coco_dict["annotations"] = filtered_annotations

    # Create the set of annotations that must exist on the images to just use the images that haver the must-have
    # annotations
    category_ids_to_filter = set(categories_to_filter.keys())
    images_filtered_annotations = [annotation for annotation in coco_dict["annotations"] if
                            str(annotation["category_id"]) in category_ids_to_filter]

    # Set of image IDs to filter
    filtered_image_ids = set([annotation["image_id"] for annotation in images_filtered_annotations])

    # Update the images if image id now is in filtered_image_ids delete the image
    image_ids_to_remove = []
    for image in tqdm.tqdm(coco_dict["images"], desc="Updating file names"):
        if image["id"] in filtered_image_ids:
            original_file_name = image["file_name"]
            # Get the file name without the path
            image_file_name = original_file_name.split("/")[-1]
            images_folder = original_file_name.split("/")[-2]
            path_to_image = os.path.join(new_image_path, images_folder, image_file_name)
            # Fix path to linux
            path_to_image = path_to_image.replace("\\", "/")
            image["file_name"] = path_to_image
        else:
            image_ids_to_remove.append(image["id"])

    # Remove the images im the image_ids_to_remove list
    coco_dict["images"] = [image for image in coco_dict["images"] if image["id"] not in image_ids_to_remove]

    # Fix the categories IDs in the annotations
    for annotation in tqdm.tqdm(coco_dict["annotations"], desc="Fixing categories IDs"):
        annotation["category_id"] = int(env.categories_transformations[str(annotation["category_id"])])

    # Save the changes in a new file in the new_file_path not exists create it
    print("Saving the new file...")
    if not os.path.exists(os.path.dirname(new_file_path)):
        os.makedirs(os.path.dirname(new_file_path))

    with open(new_file_path, 'w') as file:
        json.dump(coco_dict, file)


def split_annotations(file_path, split_file):
    """
    This function create the split file for the annotations using the split_file as a template which is a json that
    contains the keys train, val and test with the list of the document names which each image belongs to.
    :param file_path: path to the annotations file
    :param split_file: path to the split file
    :return:
    """

    # Leer el archivo JSON
    print("Reading JSON file...")
    with open(file_path, 'r') as file:
        coco_dict = json.load(file)

    # Leer el archivo de split
    print("Reading split file...")
    with open(split_file, 'r') as file:
        split_dict = json.load(file)

    # We iterate over each split set and create the respective annotations json files
    for split in split_dict:
        split_json = {}

        # Filter the images and annotations based on the split
        split_images = []
        for doc in tqdm.tqdm(split_dict[split], desc=f"Creating {split} split"):
            doc_name = doc.split(".")[0]
            split_images += [image for image in coco_dict["images"] if doc_name in os.path.basename(os.path.dirname(image["file_name"]))]

        split_image_ids = [image["id"] for image in split_images]
        split_json["images"] = split_images

        split_json["categories"] = coco_dict["categories"]

        # Filter the annotations based on the split
        split_json["annotations"] = [annot for annot in coco_dict["annotations"] if annot["image_id"] in split_image_ids]

        split_json["info"] = coco_dict["info"]

        # Save the split json
        split_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split(".")[0] + f"_{split}.json")
        with open(split_path, 'w') as file:
            json.dump(split_json, file)


def add_augmentations(file_path):
    """
    We have several augmentation archetypes, and you can modify them in the env.py file, to create the new
    sample based in the augmentation we just need to change the image_file_name to add the archetype id to
    its name. We have to set a new image for each augmentation archetype and set a new bbox for each that
    match with the bbox of the original ima
    :param file_path: path to the annotations file
    :return:
    """
    # Leer el archivo JSON
    print("Reading JSON file...")
    with open(file_path, 'r') as file:
        coco_dict = json.load(file)

    # Get the max image id and max annotation id
    max_image_id = max([image["id"] for image in coco_dict["images"]])
    max_annotation_id = max([annotation["id"] for annotation in coco_dict["annotations"]])

    # Add the augmented images to the images list
    aug_images = []
    aug_annotations = []
    for image in tqdm.tqdm(coco_dict["images"], desc="Adding augmentations"):
        original_file_name = image["file_name"]
        # Get the file name without the path
        image_file_name = os.path.basename(original_file_name)
        images_folder = os.path.dirname(original_file_name)

        # Get annotations for the image
        img_annotations = [annot for annot in coco_dict["annotations"] if annot["image_id"] == image["id"]]

        # We have several augmentation archetypes, and you can modify them in the env.py file, to create the new
        # sample based in the augmentation we just need to change the image_file_name to add the archetype id to
        # its name. We have to set a new image for each augmentation archetype and set a new bbox for each that
        # match with the bbox of the original image.
        for i in env.augmentations_archetype:
            max_image_id += 1
            path_to_aug_image = os.path.join(images_folder + env.augmentations_archetype[i], image_file_name)

            # Fix path to linux
            path_to_aug_image = path_to_aug_image.replace("\\", "/")

            new_image = {
                "id": max_image_id,
                "file_name": path_to_aug_image,
                "width": image["width"],
                "height": image["height"]
            }
            aug_images.append(new_image)

            # Add the new bboxes for the augmentation which match with the original image's bboxes. Annotation
            # format
            for annot in img_annotations:
                max_annotation_id += 1
                new_annot = {
                    "id": max_annotation_id,
                    "image_id": max_image_id,
                    "category_id": annot["category_id"],
                    "segmentation": annot["segmentation"],
                    "bbox": annot["bbox"],
                    "ignore": annot["ignore"],
                    "iscrowd": annot["iscrowd"],
                    "area": annot["area"]
                }
                aug_annotations.append(new_annot)

    # Add the new images and annotations to the coco_dict
    coco_dict["images"] += aug_images
    coco_dict["annotations"] += aug_annotations

    # Save the changes in a new file in the new_file_path not exists create it
    print("Saving the new file...")
    save_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split(".")[0] + "_augmented.json")
    with open(save_path, 'w') as file:
        json.dump(coco_dict, file)


def main():
    # Define the allowed categories
    categories = env.cartegories_to_filter
    filter_categories = env.categories_to_train

    # Define the annotations file to customize
    annotations_path = env.annotations_path

    # Define the new path for the images
    new_image_path = env.new_image_path

    # Define the save path
    new_file_path = env.new_file_path

    # Get the file name to augment  path
    augment_file = env.augment_file

    # Filter the categories and update the file names
    filter_categories_and_update_filenames(annotations_path, categories, filter_categories, new_image_path, new_file_path)

    # Split the annotations
    split_annotations(new_file_path, "./dataset/dataset_publaynet/split.json")

    # Add augmentations
    add_augmentations(augment_file)


if __name__ == '__main__':
    main()
