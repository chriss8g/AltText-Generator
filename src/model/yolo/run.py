import os
from ultralytics import YOLO
from PIL import Image

def generate_alt_texts(image_folder, model_path="yolov8x.pt", output_file="alt_texts.txt"):
    """
    Process all images in a folder using YOLO and generate alt texts based on detected objects.

    :param image_folder: Path to the folder containing images.
    :param model_path: Path to the YOLO model file (default: yolov8n.pt).
    :param output_file: Path to the output file for saving alt texts.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Verify that the image folder exists
    if not os.path.exists(image_folder):
        print(f"Error: Folder '{image_folder}' does not exist.")
        return

    # Open output file
    with open(output_file, "w") as f:
        # Loop through all files in the folder
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)

            # Check if the file is an image
            try:
                img = Image.open(file_path)
                img.verify()  # Verify that it's a valid image
            except Exception as e:
                print(f"Skipping '{filename}': {e}")
                continue

            # Run YOLO inference
            results = model(file_path)

            # Collect detected objects
            detected_objects = []
            for result in results:
                for box in result.boxes:
                    detected_objects.append(model.names[int(box.cls)])

            # Generate alt text
            if detected_objects:
                alt_text = f"Image '{filename}' contains: " + ", ".join(detected_objects) + "."
            else:
                alt_text = f"Image '{filename}' contains no recognizable objects."

            # Write to output file
            f.write(alt_text + "\n")
            print(alt_text)

# Example usage
if __name__ == "__main__":
    image_folder_path = "../../dataset/samples"  # Replace with the path to your image folder
    generate_alt_texts(image_folder_path)
