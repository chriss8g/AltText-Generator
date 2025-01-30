from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image

# Cargar el modelo y el tokenizador
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Cargar la imagen
image = Image.open("../image/ohcfh_fs/ohcfh_fs_000000709.jpg").convert("RGB")

# Procesar la imagen y generar el texto alternativo
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
output_ids = model.generate(pixel_values)

# Decodificar la salida
texto_alternativo = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(texto_alternativo)