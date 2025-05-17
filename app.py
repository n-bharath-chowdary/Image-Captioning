import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import tkinter as tk
from tkinter import filedialog
import pyttsx3 as tts

# Initialize Text-to-Speech (TTS)
engine = tts.init()



def speak(text):
    engine.setProperty("voice", engine.getProperty('voices')[0].id)
    engine.say(text)
    engine.runAndWait()


# Load BLIP Model and Processor
def load_model():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
        print("⚠️ Model loading failed:", e)



# Select Image File
def select_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    root.destroy()

    if file_path:
        return file_path
    else:
        print("❌ No image selected.")
        return None


# Generate Image Caption
def generate_caption(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    print("\n📸 Image Caption:", caption)
    speak(caption)  # Speak the generated caption


# Ensure safe multiprocessing execution on Windows
if __name__ == "__main__":
    processor, model = load_model()
    image_path = select_image()

    if image_path:
        generate_caption(image_path, processor, model)
