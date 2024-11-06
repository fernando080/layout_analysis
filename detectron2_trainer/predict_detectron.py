import os
import numpy as np
import cv2
import requests
from layoutparser.io import load_pdf
from layoutparser.visualization import draw_box
from layoutparser.models import Detectron2LayoutModel

from env import path_detectron2_train_output


def load_layout_parser_model():
    """
    Load the LayoutParser model for the extraction process.
    :return: LayoutParser model
    """

    # Load the LayoutParser model
    model = Detectron2LayoutModel(f"{path_detectron2_train_output}/config.yaml",
                                  f"{path_detectron2_train_output}/model_0059999.pth",
                                  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                  label_map={0: "CATEGORY_B", 1: "CATEGORY_C", 2: "CATEGORY_G", 3: "CATEGORY_N", 4: "CATEGORY_O", 5: "CATEGORY_P", 6: "CATEGORY_Q"})

    return model


def create_output_directory(pdf_name, output_dir):
    """
    Create the necessary directory structure in ./data/output for a specific PDF.
    This includes a folder with the name of the PDF file, and within it, the
    original_images and predicted_images folders.
    :param pdf_name: name of the PDF file
    :param output_dir: output base directory
    :return:
    """
    # Create the output directory for the PDF
    pdf_dir = os.path.join(output_dir, pdf_name)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)

    # Create the original_images directory
    original_images_dir = os.path.join(output_dir, pdf_name, 'original_images')
    if not os.path.exists(original_images_dir):
        os.makedirs(original_images_dir)

    # Create the predicted_images directory
    predicted_images_dir = os.path.join(output_dir, pdf_name, 'predicted_images')
    if not os.path.exists(predicted_images_dir):
        os.makedirs(predicted_images_dir)

    # Crete the directory containing the json file with the information from the extracted text for the original and
    # translated text
    extracted_text_dir = os.path.join(output_dir, pdf_name, 'extracted_text')
    if not os.path.exists(extracted_text_dir):
        os.makedirs(extracted_text_dir)

    return pdf_dir, original_images_dir, predicted_images_dir, extracted_text_dir


def predict_layout(image_pil, model, save_path):
    """
    It uses LayoutParser to predict the layouts of each PDF page converted to an
    image and saves the images with the layout annotations in the corresponding
    predicted_images folder.
    :param pdf_image: path to the PDF image
    :param model: LayoutParser model
    :return:
    """

    # Predict the layout
    layout = model.detect(image_pil)

    # Draw the layout over the image
    color_map = {
        'CATEGORY_B': 'red',
        'CATEGORY_C': 'blue',
        'CATEGORY_G': 'pink',
        'CATEGORY_N': 'green',
        'CATEGORY_O': 'purple',
        'CATEGORY_P': 'yellow',
        'CATEGORY_Q': 'orange',
    }
    layout_image_pil = draw_box(image_pil, layout, box_width=2, color_map=color_map)

    # Transform the PIL image to a numpy object to be used in CV2
    layout_image_np = np.array(layout_image_pil)
    # OpenCV waits BGT, so we convert from RGB to BGR
    layout_image_np = cv2.cvtColor(layout_image_np, cv2.COLOR_RGB2BGR)

    # Save the image
    cv2.imwrite(save_path, layout_image_np)

    return save_path, layout


def predict_pdf(pdf_path, ori_images_dir, pred_images_dir, extracted_text_dir):
    """
    This fucntion process the pdf file extracting the pages and predicting the layout of each page. Then saves the
    original images, the predicted images and the extracted text in the corresponding directories.
    :param pdf_path: path to the pdf file
    :param ori_images_dir: path to the directory where the original images will be saved
    :param pred_images_dir: path to the directory where the predicted images will be saved
    :param extracted_text_dir: path to the directory where the extracted text will be saved
    :return:
    """
    # Load the PDF file
    pdf = load_pdf(pdf_path, load_images=True)
    # The object returned by load_pdf is a list of PIL images, one for each page. Has the next extructure:
    # ([Layout(_blocks=[], page_data={'width': 596, 'height': 843, 'index': 0}),
    # Layout(_blocks=[], page_data={'width': 596, 'height': 843, 'index': 1}),
    # Layout(_blocks=[], page_data={'width': 596, 'height': 843, 'index': 2}),
    # ],
    # [<PIL.PpmImagePlugin.PpmImageFile image mode=RGB size=596x843 at 0x1F26FF194E0>,
    # <PIL.PpmImagePlugin.PpmImageFile image mode=RGB size=596x843 at 0x1F26FF19510>,
    # <PIL.PpmImagePlugin.PpmImageFile image mode=RGB size=596x843 at 0x1F26FF734F0>,
    # ])

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Predict the layout of each page.
    model = load_layout_parser_model()
    for i, page_image in enumerate(pdf[1]):
        # Save the original image
        original_image_path = os.path.join(ori_images_dir, f'{pdf_name}_p{i}.png')
        print("original_image: ", original_image_path)
        page_image.save(original_image_path)

        # Predict the layout
        save_path = os.path.join(pred_images_dir, f'{pdf_name}_p{i}_layout.png')
        predict_layout(page_image, model, save_path)


def prefict_all_pdfs(pdf_dir, output_dir):
    """
    Predict the layout of all the PDFs in the directory.
    :param pdf_dir: path to the directory containing the PDFs
    :param ori_images_dir: path to the directory where the original images will be saved
    :param pred_images_dir: path to the directory where the predicted images will be saved
    :param extracted_text_dir: path to the directory where the extracted text will be saved
    :return:
    """
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        pdf_dir, ori_images_dir, pred_images_dir, extracted_text_dir = create_output_directory(pdf_name, output_dir)
        predict_pdf(pdf_file, ori_images_dir, pred_images_dir, extracted_text_dir)


def main():
    print("Content of the current directory:")
    print(os.listdir('.'))
    print("Content of the ./test_pdfs directory:")
    print(os.listdir('./test_pdfs'))

    pdf_path = r'./test_pdfs'
    output_dir = r'./output_pdfs_detect'

    # Predict the layout of all the PDFs in the directory
    prefict_all_pdfs(pdf_path, output_dir)


if __name__ == '__main__':
    main()
