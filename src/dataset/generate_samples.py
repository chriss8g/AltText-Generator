import random
import os
import requests

# Archivo que contiene los nombres de los archivos
file_list = "list.txt"

# Carpeta donde se guardarán las imágenes
output_folder = "samples"
os.makedirs(output_folder, exist_ok=True)

# URL base
base_url = "https://iiif.ohc.cu/iiif/3/{}/full/max/0/default.jpg"

# Leer todos los nombres del archivo
with open(file_list, "r") as file:
    lines = [line.strip() for line in file.readlines()]

# Seleccionar 30 nombres al azar
random_files = random.sample(lines, 30)

# Descargar las imágenes
for filename in random_files:
    url = base_url.format(filename)
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Guardar la imagen
            image_path = os.path.join(output_folder, f"{filename}")
            with open(image_path, "wb") as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            print(f"Imagen guardada: {image_path}")
        else:
            print(f"No se pudo descargar la imagen para {filename} (Código HTTP {response.status_code})")
    except Exception as e:
        print(f"Error al descargar {filename}: {e}")
