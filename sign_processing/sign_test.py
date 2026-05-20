import os
import cv2
# Import the engine logic from your separate engine file
from sign import PureSignatureValidator

if __name__ == "__main__":
    # Get the exact directory path where this script file lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Supported image extensions
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    
    # List all files in this script's directory
    all_files = os.listdir(current_dir)
    
    # Filter to only pick original images (skips python scripts and pipeline outputs)
    image_files = [
        f for f in all_files 
        if f.lower().endswith(valid_extensions) and not f.startswith("output_")
    ]

    if not image_files:
        print(f"No original images found in: {current_dir}")
        print("Drop some images into this exact folder to test the pipeline.")
        exit()

    print(f"Found {len(image_files)} image(s) to verify in the folder.\n" + "-" * 60)

    # CRITICAL FIX: Instantiate the validator class object once before the loop
    # This prevents the AI model weights from reloading on every single image
    validator_engine = PureSignatureValidator()

    # Process each image file
    for file_name in image_files:
        # Construct path for the current file
        input_path = os.path.join(current_dir, file_name)
        print(f"\nProcessing File: {file_name}")
        
        # FIX: Call the method on the instantiated object instance
        success, message, output_img = validator_engine.verify_and_process(input_path)
        
        print(f"-> Verification Status : {success}")
        print(f"-> Engine Message      : {message}")
        
        if success:
            # Save the output image inside this same directory with an 'output_' prefix
            output_name = f"output_{file_name}"
            output_path = os.path.join(current_dir, output_name)
            
            cv2.imwrite(output_path, output_img)
            print(f"-> Verification Output : Saved verified file to {output_name}")
        else:
            print("-> Verification Output : REJECTED. Failed verification logic rules.")
