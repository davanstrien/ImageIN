import argparse
import gradio as gr

from PIL import Image
from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor

HF_MODEL_PATH = (
    "ImageIN/convnext-base-224_finetuned_on_unlabelled_IA_with_snorkel_labels"
)


def read_image_pil_file(image_file) -> Image:
    with Image.open(image_file) as image:
        return image.convert(mode=image.mode)


def get_prediction(filename: str, pipeline: pipeline) -> str:
    result = pipeline(read_image_pil_file(filename))
    return result[0]["label"]


classif_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_PATH)
feature_extractor = AutoFeatureExtractor.from_pretrained(HF_MODEL_PATH)

classif_pipeline = pipeline(
    "image-classification", model=classif_model, feature_extractor=feature_extractor
)

parser = argparse.ArgumentParser(
    description="Script to classify image as illustrated page or not"
)
parser.add_argument("--filename", type=str)
args = parser.parse_args()

# prediction from a local image file
if args.filename:
    print(get_prediction(args.filename, classif_pipeline))

demo = gr.Interface(
    fn=lambda x: classif_pipeline(x)[0]["label"],
    inputs=gr.Image(type="pil"),
    outputs="text",
)
demo.launch(share=False)

print("__DONE__")
