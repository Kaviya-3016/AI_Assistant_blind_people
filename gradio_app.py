import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import numpy as np
import tempfile
from gtts import gTTS
from deep_translator import GoogleTranslator

# Optional imports (install in requirements)
try:
    from ultralytics import YOLO
except:
    YOLO = None

try:
    import pytesseract
except:
    pytesseract = None

# Lazy globals
custom_model = None
blip_processor = None
blip_model = None


def load_models():
    global custom_model, blip_processor, blip_model
    if custom_model is None and YOLO is not None:
        try:
            custom_model = YOLO('best.pt')
        except:
            custom_model = None
    if blip_processor is None or blip_model is None:
        try:
            blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
            blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        except:
            blip_processor = None
            blip_model = None


def make_tts(text, language):
    if not text:
        return None
    lang = 'ta' if language == 'Tamil' else 'en'
    if language == 'Tamil':
        try:
            text = GoogleTranslator(source='auto', target='ta').translate(text)
        except:
            pass
    path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
    tts = gTTS(text=text, lang=lang)
    tts.save(path)
    return path


def process(img, language, mode):
    if img is None:
        return 'Please upload an image.', None, None

    load_models()
    result_lines = []

    # OCR
    if pytesseract is not None:
        try:
            text = pytesseract.image_to_string(img)
            if text.strip():
                result_lines.append('OCR Text: ' + text[:500])
        except:
            pass

        # BLIP Captioning
    if blip_processor is not None and blip_model is not None:
        try:
            inputs = blip_processor(images=img, return_tensors='pt')
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_length=40)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            result_lines.append('Caption: ' + caption)
        except:
            pass

    # YOLO
    annotated = img
    if custom_model is not None:
        try:
            results = custom_model(np.array(img))
            plotted = results[0].plot()
            annotated = Image.fromarray(plotted[..., ::-1])
            names = []
            for c in results[0].boxes.cls.tolist():
                names.append(custom_model.names[int(c)])
            if names:
                result_lines.append('Detected: ' + ', '.join(sorted(set(names))))
        except:
            pass

    if custom_model is None:
        result_lines.append('Custom model best.pt not found. Put best.pt in the same folder as this app.')

    if not result_lines:
        result_lines.append('Image processed successfully.')

    final_text = '\n\n'.join(result_lines)
    audio = None if mode == 'Silent Mode' else make_tts(final_text, language)
    return final_text, annotated, audio


with gr.Blocks(title='VisionAid AI') as demo:
    gr.Markdown('# VisionAid AI')
    gr.Markdown('AI Assistant for Visually Impaired Users')

    with gr.Row():
        language = gr.Dropdown(['English', 'Tamil'], value='English', label='Language')
        mode = gr.Dropdown(['Scene Description', 'Silent Mode'], value='Scene Description', label='Mode')

    image = gr.Image(type='pil', label='Upload or Capture Image', sources=['upload', 'webcam'])
    btn = gr.Button('Process', variant='primary')

    out_text = gr.Textbox(label='Result', lines=10)
    out_image = gr.Image(label='Annotated Image')
    out_audio = gr.Audio(label='Audio Output', type='filepath')

    btn.click(process, inputs=[image, language, mode], outputs=[out_text, out_image, out_audio])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
