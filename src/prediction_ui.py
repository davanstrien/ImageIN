import gradio as gr

from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor

HF_MODEL_PATH = (
    "ImageIN/convnext-base-224_finetuned_on_unlabelled_IA_with_snorkel_labels"
)

classif_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_PATH)
feature_extractor = AutoFeatureExtractor.from_pretrained(HF_MODEL_PATH)

classif_pipeline = pipeline(
    "image-classification", model=classif_model, feature_extractor=feature_extractor
)

OUTPUT_SENTENCE = "This image is {result}."


def get_formatted_prediction(img) -> str:
    return OUTPUT_SENTENCE.format(
        result=classif_pipeline(img)[0]["label"].replace("-", " ")
    )


demo = gr.Interface(
    fn=get_formatted_prediction,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="ImageIN",
    description="Identify illustrations in pages of historical books!",
    examples=["old_book_page.png", "women_book_image.png", "page_with_images.png"],
)
demo.launch()
