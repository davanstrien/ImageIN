import bentoml

from bentoml.io import Image, JSON
from transformers import pipeline

from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class PretrainedModelRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.extractor = AutoFeatureExtractor.from_pretrained(
            "ImageIN/convnext-base-224_finetuned_on_ImageIn_annotations"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "ImageIN/convnext-base-224_finetuned_on_ImageIn_annotations"
        )
        self.unmasker = pipeline(
            task="image-classification",
            model=self.model,
            feature_extractor=self.extractor,
        )

    @bentoml.Runnable.method(batchable=False)
    def __call__(self, input_image):
        return self.unmasker(input_image)


runner = bentoml.Runner(PretrainedModelRunnable, name="pretrained_image_classifier")

svc = bentoml.Service("pretrained_image_classifier", runners=[runner])


@svc.api(input=Image(), output=JSON())
def unmask(input_series: str) -> list:
    return runner.run(input_series)
