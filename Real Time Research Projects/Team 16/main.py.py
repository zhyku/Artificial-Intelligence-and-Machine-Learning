
import argparse
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os

# Load model and supporting components
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configuration
default_max_length = 16
default_num_beams = 4
gen_kwargs = {"max_length": default_max_length, "num_beams": default_num_beams}

# Prediction function
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# Main CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for input images.")
    parser.add_argument('--images', nargs='+', required=True, help="Paths to image files")
    args = parser.parse_args()

    results = predict_step(args.images)
    for img, caption in zip(args.images, results):
        print(f"{os.path.basename(img)}: {caption}")
