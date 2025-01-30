import os
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    MarianMTModel, MarianTokenizer,
    VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer,
    CLIPProcessor, CLIPModel
)
from PIL import Image
import pandas as pd


def is_image_processed(csv_path, filename):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return filename in df['image_name'].tolist()
    return False

# Cargar modelos BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Cargar modelo de traducción (Inglés a Español)
translator_model_name = "Helsinki-NLP/opus-mt-en-es"
translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator_model = MarianMTModel.from_pretrained(translator_model_name)

# Cargar modelo ViT-GPT2
vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_gpt2_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_gpt2_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Cargar modelo CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_blip_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def generate_vit_gpt2_caption(image):
    pixel_values = vit_gpt2_feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = vit_gpt2_model.generate(pixel_values)
    caption = vit_gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def translate_to_spanish(text):
    translated = translator_model.generate(**translator_tokenizer(text, return_tensors="pt", padding=True))
    caption_spanish = translator_tokenizer.decode(translated[0], skip_special_tokens=True)
    return caption_spanish

def compare_image_descriptions(image, description1, description2):
    inputs = clip_processor(text=[description1, description2], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    similarity_scores = probs[0].tolist()
    best_match_index = similarity_scores.index(max(similarity_scores))
    best_description = description1 if best_match_index == 0 else description2
    return best_match_index, similarity_scores[best_match_index], best_description

def process_folder(mother_folder):
    
    for root, dirs, files in os.walk(mother_folder):
        if root == mother_folder:
            continue
        csv_path = os.path.join(root, "results.csv")
        
        print(f"\nProcesando carpeta: {root}")
        results = []
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                if is_image_processed(csv_path, file):
                    print("Image: " + file + " already processed")
                    continue
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert("RGB")
                
                print(f"  Procesando imagen: {file} con BLIP...")
                blip_caption = generate_blip_caption(image)
                
                print(f"  Procesando imagen: {file} con ViT-GPT2...")
                vit_gpt2_caption = generate_vit_gpt2_caption(image)
                
                print(f"  Comparando descripciones con CLIP...")
                best_match_index, best_score, best_description = compare_image_descriptions(image, blip_caption, vit_gpt2_caption)
                
                print(f"  Traduciendo descripción seleccionada...")
                best_description_spanish = translate_to_spanish(best_description)
                
                # Guardar resultados
                results.append({
                    "image_name": file,
                    "blip_caption": blip_caption,
                    "vit_gpt2_caption": vit_gpt2_caption,
                    "best_model": best_match_index,
                    "best_score": best_score,
                    "best_description_spanish": best_description_spanish
                })
        
        # Guardar resultados en un CSV dentro de la carpeta correspondiente
        if results:
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            print(f"  Resultados guardados en: {csv_path}")


mother_folder = "../images/" 
process_folder(mother_folder)