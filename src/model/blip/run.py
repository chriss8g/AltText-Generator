import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image

# Rutas de entrada y salida
input_folder = "../../dataset/samples"  # Carpeta con las imágenes
output_file = "descripciones.txt"  # Archivo de salida

# Cargar el modelo y procesador BLIP
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Cargar el modelo de traducción (Inglés a Español)
translator_model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator = MarianMTModel.from_pretrained(translator_model_name)

# Crear o limpiar el archivo de salida
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Imagen\tDescripción\n")  # Encabezados opcionales

# Procesar todas las imágenes de la carpeta
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(input_folder, filename)
        
        try:
            # Cargar la imagen
            image = Image.open(image_path).convert("RGB")

            # Preparar la imagen para el modelo
            inputs = processor(images=image, return_tensors="pt")

            # Generar la descripción
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # Traducir la descripción
            translated = translator.generate(**tokenizer(caption, return_tensors="pt", padding=True))
            caption_spanish = tokenizer.decode(translated[0], skip_special_tokens=True)

            # Escribir en el archivo de salida
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{filename}\t{caption_spanish}\n")

            print(f"Procesado: {filename} -> {caption_spanish}")
        
        except Exception as e:
            print(f"Error procesando {filename}: {e}")
