import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def interface_function(image):
    if image is None:
        return None, "Brak obrazu."
    caption = generate_caption(image)
    return image, caption

def save_caption(caption):
    filename = "opis_obrazu.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(caption)
    return f"Opis zapisano do pliku: {filename}"

def editable_interface(image):
    img, caption = interface_function(image)
    return img, caption

with gr.Blocks(title="AI Opisywanie Obrazów") as demo:
    gr.Markdown("""
    # Opisywanie Obrazów z AI
    Prześlij obraz (JPG/PNG), a AI wygeneruje jego opis. Możesz go edytować i zapisać.
    """)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Prześlij obraz")
        image_output = gr.Image(type="pil", label="Podgląd obrazu")

    caption_textbox = gr.Textbox(label="Opis obrazu", lines=3)

    with gr.Row():
        gen_button = gr.Button("Generuj opis")
        save_button = gr.Button("Zapisz opis")

    gen_button.click(editable_interface, inputs=image_input, outputs=[image_output, caption_textbox])
    save_button.click(save_caption, inputs=caption_textbox, outputs=caption_textbox)

demo.launch()
