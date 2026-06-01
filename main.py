import sys
import os
import cv2
import customtkinter as ctk
from tkinter import filedialog

import numpy as np

# Force clean light theme styles globally
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Import Core Processing Pipelines
from face_processing.pipeline import FaceProcessingPipeline
from sign_processing.sign import PureSignatureValidator

class ImageProcessingStudio(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Window Geometry Setup ---
        self.title("Passport Document Processor")
        self.geometry("900x700")
        self.configure(fg_color="#F4F6F9")  # Off-white background canvas

        # --- Core Logic Instantiations ---
        self.face_processor = FaceProcessingPipeline()
        self.sign_processor = PureSignatureValidator()

        # --- Central White Base Dashboard Frame Card ---
        self.card = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=15, border_width=1, border_color="#E2E8F0")
        self.card.pack(fill="both", expand=True, padx=40, pady=40)

        # --- 1. Header Typography Layout Section ---
        self.title_label = ctk.CTkLabel(
            self.card, 
            text="Image Processing Studio", 
            font=("Arial", 28, "bold"), 
            text_color="#1E293B"
        )
        self.title_label.pack(pady=(40, 5))

        self.subtitle_label = ctk.CTkLabel(
            self.card, 
            text="Convert photos to passport size and enhance signatures", 
            font=("Arial", 14), 
            text_color="#64748B"
        )
        self.subtitle_label.pack(pady=(0, 30))

        # --- 2. Input Directory Selection Elements ---
        self.input_label = ctk.CTkLabel(
            self.card, 
            text="📁  Input Folder", 
            font=("Arial", 15, "bold"), 
            text_color="#1E293B"
        )
        self.input_label.pack(anchor="w", padx=40, pady=(10, 5))

        self.input_row = ctk.CTkFrame(self.card, fg_color="transparent")
        self.input_row.pack(fill="x", padx=40, pady=(0, 15))

        self.input_entry = ctk.CTkEntry(
            self.input_row, 
            height=45, 
            fg_color="#F1F5F9", 
            text_color="#334155", 
            border_color="#CBD5E1"
        )
        self.input_entry.pack(side="left", fill="x", expand=True)

        self.input_btn = ctk.CTkButton(
            self.input_row, 
            text="📤 Browse", 
            width=120, 
            height=45, 
            fg_color="#0F172A", 
            hover_color="#1E293B", 
            text_color="#FFFFFF", 
            font=("Arial", 14, "bold"),
            command=self.browse_input
        )
        self.input_btn.pack(side="right", padx=(15, 0))

        # --- 3. Output Directory Selection Elements ---
        self.output_label = ctk.CTkLabel(
            self.card, 
            text="📥  Base Output Folder", 
            font=("Arial", 15, "bold"), 
            text_color="#1E293B"
        )
        self.output_label.pack(anchor="w", padx=40, pady=(10, 5))

        self.output_row = ctk.CTkFrame(self.card, fg_color="transparent")
        self.output_row.pack(fill="x", padx=40, pady=(0, 20))

        self.output_entry = ctk.CTkEntry(
            self.output_row, 
            height=45, 
            fg_color="#F1F5F9", 
            text_color="#334155", 
            border_color="#CBD5E1"
        )
        self.output_entry.pack(side="left", fill="x", expand=True)

        self.output_btn = ctk.CTkButton(
            self.output_row, 
            text="📁 Browse", 
            width=120, 
            height=45, 
            fg_color="#0F172A", 
            hover_color="#1E293B", 
            text_color="#FFFFFF", 
            font=("Arial", 14, "bold"),
            command=self.browse_output
        )
        self.output_btn.pack(side="right", padx=(15, 0))

        # --- 4. Dynamic Inline Alert Notification Block ---
        self.status_box = ctk.CTkFrame(self.card, height=65, fg_color="#F8FAFC", corner_radius=8, border_width=1, border_color="#E2E8F0")
        self.status_box.pack(fill="x", padx=40, pady=15)
        self.status_box.pack_propagate(False)

        self.status_text = ctk.CTkLabel(
            self.status_box, 
            text="Ready. Please select your folders to begin sorting.", 
            font=("Arial", 13, "normal"), 
            text_color="#475569"
        )
        self.status_text.pack(side="left", padx=20, fill="both")

        # --- 5. Primary Start Action Execution Control Button ---
        self.start_btn = ctk.CTkButton(
            self.card, 
            text="Start Processing", 
            height=50, 
            fg_color="#0F172A", 
            hover_color="#1E293B", 
            text_color="#FFFFFF", 
            font=("Arial", 15, "bold"),
            command=self.run_pipeline
        )
        self.start_btn.pack(fill="x", padx=40, pady=(15, 40))

    def browse_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, folder)
            if not self.output_entry.get():
                self.output_entry.insert(0, os.path.join(folder, "Processed_Studio_Output"))

    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, folder)

    def run_pipeline(self):
        """Main processing flow containing intelligent branch-routing rules."""
        input_folder = self.input_entry.get().strip()
        base_output = self.output_entry.get().strip()

        if not input_folder or not base_output:
            self.update_status("⚠️ Please select valid input and output directory paths.", "#FEF2F2", "#EF4444", "#DC2626")
            return

        # Explicit Sub-Folder Branch Generation
        face_folder = os.path.join(base_output, "faces")
        sign_folder = os.path.join(base_output, "signatures")
        error_folder = os.path.join(base_output, "errors")

        for folder in [face_folder, sign_folder, error_folder]:
            os.makedirs(folder, exist_ok=True)

        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_extensions)]

        if not files:
            self.update_status("⚠️ No valid image files found in the source directory.", "#FFFBEB", "#F59E0B", "#B45309")
            return

        self.start_btn.configure(state="disabled", text="Processing & Sorting Images...")

        for idx, file_name in enumerate(files, 1):
            self.update_status(f"⏳ Classifying and Processing [{idx}/{len(files)}]: {file_name}", "#F8FAFC", "#E2E8F0", "#475569")
            file_path = os.path.join(input_folder, file_name)

            try:
                image = cv2.imread(file_path)
                if image is None:
                    # Unreadable file corruption fallback
                    cv2.imwrite(os.path.join(error_folder, file_name), cv2.imread(file_path) if cv2.imread(file_path) is not None else np.zeros((100,100,3)))
                    continue

                # --- STEP 1: PRE-SCREEN WITH FACE DETECTION LAYER ---
                face_bbox = self.face_processor.detector.detect_face(image)
                
                if face_bbox is not None:
                    # It's an image containing a face. Route exclusively to the face pipeline.
                    face_result = self.face_processor.process_image(image)
                    if face_result is not None:
                        cv2.imwrite(os.path.join(face_folder, file_name), face_result)
                    else:
                        # Face detection found a face box, but cropping/background formatting steps failed
                        cv2.imwrite(os.path.join(error_folder, f"failed_face_{file_name}"), image)
                    
                    self.update()
                    continue  # Successfully handled as a face image. Skip to next file.

                # --- STEP 2: ALTERNATE ROUTE TO SIGNATURE VALIDATOR ---
                success, log_msg, sign_result = self.sign_processor.verify_and_process(file_path)
                
                if success and sign_result is not None and sign_result.size > 0:
                    # It's a valid signature document. Save to signatures folder.
                    cv2.imwrite(os.path.join(sign_folder, file_name), sign_result)
                else:
                    # Failed both face and signature structure verification layers -> Send to errors directory
                    cv2.imwrite(os.path.join(error_folder, file_name), image)

                self.update()

            except Exception as e:
                print(f"Exception handling {file_name}: {str(e)}")
                # Any runtime execution failure dumps the original image into the errors bucket safely
                try:
                    cv2.imwrite(os.path.join(error_folder, f"broken_{file_name}"), cv2.imread(file_path))
                except:
                    pass

        self.update_status("✅ Sorting Complete!\nFiles split clean into /faces, /signatures, and /errors directories.", "#F0FDF4", "#DCFCE7", "#15803D")
        self.start_btn.configure(state="normal", text="Start Processing")

    def update_status(self, message, bg_color, border_color, text_color):
        self.status_box.configure(fg_color=bg_color, border_color=border_color)
        self.status_text.configure(text=message, text_color=text_color)
        self.update()

if __name__ == "__main__":
    app = ImageProcessingStudio()
    app.mainloop()
