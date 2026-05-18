import os
import glob
import cv2

# Assuming your pipeline code is in a file named pipeline.py
from pipeline import FaceProcessingPipeline


def run_batch_processing():
    # Initialize your pipeline
    pipeline = FaceProcessingPipeline()

    # Setup directories relative to this script
    project_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_dir, "input_images")
    output_dir = os.path.join(project_dir, "output_images")

    # Create directories if they do not exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    input_files = []
    for ext in extensions:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not input_files:
        print(f"No images found in: {input_dir}")
        print(f"Please drop some images into '{input_dir}' and run again.")
        return

    print(f"Found {len(input_files)} images to process.")

    # Load images, keeping track of filenames
    valid_images = []
    filenames = []

    for file_path in input_files:
        img = cv2.imread(file_path)
        if img is not None:
            valid_images.append(img)
            filenames.append(os.path.basename(file_path))
        else:
            print(f"Warning: Could not read {file_path}")

    # Process using the pipeline's batch function
    print("Starting batch processing...")
    # Map back manually to handle individual naming, or use process_image loop
    # Since process_batch filters out None, standard loop maps naming perfectly:
    for filename, img in zip(filenames, valid_images):
        print(f"Processing: {filename}...")
        result = pipeline.process_image(img)

        if result is not None:
            # Construct output filename (e.g., photo.jpg -> passport_photo.jpg)
            output_name = f"passport_{filename}"
            output_path = os.path.join(output_dir, output_name)
            
            # Save output image
            cv2.imwrite(output_path, result)
            print(f"-> Saved: {output_path}")
        else:
            print(f"-> Skipped/Failed processing for: {filename}")

    print("\nBatch processing completed successfully!")


if __name__ == "__main__":
    run_batch_processing()
