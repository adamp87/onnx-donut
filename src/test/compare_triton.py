import numpy as np
from PIL import Image

from onnx_donut.predictor import OnnxPredictor
from onnx_donut.predictor_triton import OnnxPredictorTriton


def infer_triton():
    # Image path to run on
    img_path = "img.jpg"

    # Folder where the exported model will be stored
    onnx_folder = "converted_donut"
    # Read image
    img = np.array(Image.open(img_path).convert('RGB'))

    # Instantiate ONNX predictor
    predictor = OnnxPredictorTriton(model_folder=onnx_folder)

    # Write your prompt accordingly to the model you use
    prompt = f"<s_docvqa><s_question>How much is the fish?</s_question><s_answer>"

    # Run prediction
    out = predictor.generate(img, prompt)

    # Display prediction
    print(out)

    # Free resources
    del predictor

def infer_onnx():
    # Image path to run on
    img_path = "img.jpg"

    # Folder where the exported model will be stored
    onnx_folder = "converted_donut"
    # Read image
    img = np.array(Image.open(img_path).convert('RGB'))

    # Instantiate ONNX predictor
    predictor = OnnxPredictor(model_folder=onnx_folder)

    # Write your prompt accordingly to the model you use
    prompt = f"<s_docvqa><s_question>How much is the fish?</s_question><s_answer>"

    # Run prediction
    out = predictor.generate(img, prompt)

    # Display prediction
    print(out)

    # Free resources
    del predictor

if __name__ == "__main__":
    infer_onnx()
    infer_triton()