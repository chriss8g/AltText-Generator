import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

def generate_caption_blip(image):
    # Cargar el modelo y procesador BLIP
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Generar descripción
    inputs = blip_processor(images=image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    caption_blip = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return caption_blip

def generate_caption_vit(image):
    # Cargar el modelo y el tokenizador
    vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Generar descripción
    pixel_values = vit_feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = vit_model.generate(pixel_values)
    caption_vit = vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption_vit

def compare_image_descriptions(image, description1, description2):
    # Cargar modelo y procesador de CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=[description1, description2], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Determinar la descripción más similar
    similarity_scores = probs[0].tolist()
    best_match_index = similarity_scores.index(max(similarity_scores))
    best_description = description1 if best_match_index == 0 else description2

    return best_description, similarity_scores

image_path = "../images/ohcjg_jg/ohcjg_jg_000000611.jpg"
image = Image.open(image_path).convert("RGB")

# Generar descripciones
caption_blip = generate_caption_blip(image)
caption_vit = generate_caption_vit(image)

# Comparar descripciones con CLIP
best_description, scores = compare_image_descriptions(image, caption_blip, caption_vit)

# Imprimir resultados
print(f"Descripción generada por BLIP: {caption_blip}")
print(f"Descripción generada por ViT-GPT2: {caption_vit}")
print(f"La descripción más similar según CLIP es: '{best_description}'")
print(f"Puntuaciones: {scores}")