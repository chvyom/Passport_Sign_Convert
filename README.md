## 📸 Automated Passport Photo & Signature Extractor
An automated desktop application designed to streamline the processing of bulk identity documents. It takes an input folder containing raw images, automatically identifies human faces and handwritten signatures, and exports them as perfectly formatted, production-ready assets.
## ✨ Key Features

* Smart Face Detection & Alignment: Automatically detects human faces within mixed images using machine learning models.
* Passport Photo Generation: Crops, centers, and scales faces to standard passport-size dimensions with clean, uniform backgrounds.
* Precision Signature Cropping: Identifies handwritten signatures in documents or images and performs high-quality, fine-edge cropping.
* Bulk Folder Processing: Processes hundreds of images sequentially from a selected input directory and saves organized outputs into dedicated folders.
* Smart Error Isolation: Separates unprocessable or low-quality files automatically, ensuring your main output folders remain completely clean.
* Zero Dependencies for Users: Compiled into a single, standalone executable (.exe) that requires no Python installation or environment setup.

## 🚀 How It Works

   1. Input: Drop your raw images into a designated source folder.
   2. Process: Run the application and select your input and output directory paths.
   3. Output: The application instantly generates organized subfolders for successful results and isolates problematic files.

## 📂 Output & Error Folder Structure
The application automatically creates the following folder structure inside your chosen output directory:

* 📁 passport_photos/ — Centered, scaled, and background-adjusted passport images.
* 📁 signatures/ — Finely cropped, clean signature images ready for use.
* 📁 errors/ — Contains original files that failed processing.
* Why do files end up here? An image moves to the error folder if the AI cannot confidently detect a face, if the signature is missing/unreadable, or if the input image file is corrupted. This allows you to manually review only the problematic files without digging through the successful ones.
