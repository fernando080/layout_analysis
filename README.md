# Layout Analysis

This project is focused on document layout analysis using [Microsoft's LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) and [Facebook's Detectron2](https://github.com/facebookresearch/detectron2) for this second case I used as the base code for the training the repository from [Layout Parser](https://github.com/Layout-Parser/layout-parser) and its repository to train on your custom dataset [Layout Parser Model Training](https://github.com/Layout-Parser/layout-model-training). The objective is to analyze and classify document layouts by training on a custom dataset and using layout-based object detection models.

**IMPORTANT**
- I added some modifications and updates to the original code to adapt it to my custom dataset and the current state of the used modules and libraries. There exists margin for improvement and optimization in the code, and I encourage you to explore and experiment with the code to enhance its performance and capabilities.
- I did not finish the script to predict using the layoutlmv3 model but I suspect that you can use the same script from detectron2.

## Table of Contents
- Project Structure
- Dataset Preparation
- Requirements
- Training the Model
- Acknowledgements

## Project Structure
The project is structured as follows:
```
layout_analysis/
│
├── dataset/
│   ├── dataset_20240703/
│   ├── images_sample/
│   ├── train_outputs/
│   └── test_pdfs/
│
├── detector2_trainer/
│   ├── configs/
│   ├── scripts/
│   ├── tools/
│   └── models/
│
├── layoutlmv3/
│   ├── examples/
│   ├── models/
│   ├── outputs/
│   └── train_outputs/
│
├── .gitignore
├── Dockerfile.detectron
├── Dockerfile.layoutlmv3
└── train.sh
```

## Dataset Preparation
The dataset preparation involves processing raw images from document pages, I have my own data with documento from publaynet annotated using my onw categories using [Label Studio](https://github.com/HumanSignal/label-studio) and I exported the data using the label studio interface.

1. **Annotations Customization**:
    - I created <customize_dataset.py> file to play with the categories and make some data engineering to keep the most relevant categories that helped me to balance the dataset.
2. **Split JSON**:
    - I have a JSON file with documents ids distributed in train, validation and test sets. I used it in <customize_dataset.py> to create the splits in different output JSONs. I let you bellow the structre:
        ```json
        {
          "train": [
            "doc1.pdf", 
            "doc2.pdf", 
            "doc3.pdf"
          ],
          "val": [
            "doc4.pdf" 
          ],
          "test": [
            "doc5.pdf"
          ]
        }
    ```
3. **User of [Layout Parser](https://github.com/Layout-Parser/layout-parser)**: 
   - I used the repository [Layout Parser Model Training](https://github.com/Layout-Parser/layout-model-training) to train the detectron2 model with my custom dataset. I used the script `train_modified.py` to train the model and the script `predict_detectron.py` and `predict_publaynet_detectron.py` to test the model.
4. **Data Augmentation**:
   - I used [Augraphy](https://github.com/sparkfish/augraphy) to augment the images.
   - The add_augmentations function can be used to add augmentations to the dataset.

## Requirements
The requirements for this project are in the `requirements.txt` file from each folder. 

## Training the Model
To train the model, you can use the `train.sh` script. This script will run the training process for both LayoutLMv3 and Detectron2 models. You can modify the script to include additional parameters or configurations as needed. Check the documentations of each repository to understand the parameters and configurations available. The documentation found in [Layout Parser Model Training](https://github.com/Layout-Parser/layout-model-training) and in [Microsoft's LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) are similar I kindly encourage you to read them.

Ensure this script invokes `train_net.py` with the following arguments:
- `--dataset_name`: The name of the dataset to be used for training.
- `--json_annotation_train`: Path to your training set annotations JSON.
- `--image_path_train`: Path to your training images.
- `--json_annotation_val`: Path to your validation set annotations JSON.
- `--image_path_val`: Path to your validation images.
- `--config-file`: Path to the configuration YAML file for LayoutLMv3.
- 
Ensure that the paths to your JSON files and images are correctly set in `train.sh` before running the script.

## Acknowledgements
This project relies on resources and code from the following:
- [Layout Parser](https://github.com/Layout-Parser/layout-parser)
- [Layout Parser Model Training](https://github.com/Layout-Parser/layout-model-training)
- [Microsoft's LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
- [Facebook's Detectron2](https://github.com/facebookresearch/detectron2)
