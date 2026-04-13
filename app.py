import streamlit as st
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from transformers import AutoProcessor, BlipForConditionalGeneration, ImageFile
from PIL import Image
from PIL import Image, UnidentifiedImageError
import pytesseract
import re
import numpy as np
import tempfile
import os
from gtts import gTTS
from io import BytesIO
from deep_translator import GoogleTranslator
import platform
import shutil

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title='VisionAid AI',page_icon='icon.png', layout='wide')
st.markdown('''<style>
.stApp{background:linear-gradient(180deg,#050b18,#02060f);color:white;}
.block-container{max-width:760px;padding-top:2rem;}
h1,h2,h3,label,p,div{color:white !important;}
.stSelectbox div[data-baseweb="select"]>div,.stTextInput input,.stFileUploader{background:#232634;border-radius:10px;}
.stButton>button{border-radius:10px;}
</style>''', unsafe_allow_html=True)
st.title('AI Assistant for Visually Impaired')

# -------------------------------
# TESSERACT CONFIG
# -------------------------------

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
CONF_THRESHOLD = 50 


language = st.selectbox('Select Language', ['English', 'Tamil'])
mode = st.selectbox('Mode Selection', ['Scene Description', 'Silent Mode'])
input_type = st.radio('Input Type', ['Upload Image', 'Live Camera'])

image = None
image_path = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
# -------------------------------
# INPUT HANDLING
# -------------------------------
if input_type == 'Upload Image':
    uploaded_file = st.file_uploader(
        'Upload an image',
        type=['jpg', 'jpeg', 'png', 'webp']
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            image.verify()

            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            image.save(tfile.name)
            image_path = tfile.name

            st.image(image, caption='Uploaded Image', use_container_width=True)

        except UnidentifiedImageError:
            st.error("Invalid image file. Please upload JPG, PNG, or WEBP.")

        except Exception as e:
            st.error(f"Error loading image: {e}")

else:
    camera_file = st.camera_input("Take a photo")

    if camera_file is not None:
        try:
            camera_file.seek(0)
            image = Image.open(camera_file).convert("RGB")

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            image.save(tfile.name)
            image_path = tfile.name

            st.image(image, caption='Captured Image', use_container_width=True)

        except Exception as e:
            st.error(f"Could not read camera image: {e}")

# -------------------------------
# PROCESS ONLY IF IMAGE EXISTS
# -------------------------------
if image is not None:
    # -------------------------------
    # ADD-ON: TEXT vs IMAGE DETECTION
    # -------------------------------
    ocr_preview = pytesseract.image_to_string(image)
    text_ratio = len(ocr_preview.strip()) / 100
    IS_TEXT_IMAGE = text_ratio > 0.2

    st.write('Text Ratio:', text_ratio)
    st.write('Is Text Image:', IS_TEXT_IMAGE)

    # ============================================================
    # ================= OCR PIPELINE (NEW) ========================
    # ============================================================
    if IS_TEXT_IMAGE:
        st.subheader('OCR OUTPUT')
        data = pytesseract.image_to_data(
            image,
            lang='eng+tam+equ',
            output_type=pytesseract.Output.DICT
        )

        text_eng, text_ta, text_formula = [], [], []
        regex_eng = r'[A-Za-z0-9.,!?]+'
        regex_tam = r'[\u0B80-\u0BFF]+'
        regex_formula = r'[=∑∫∏√±\^\-*/(){}\[\]]+'

        for i, word in enumerate(data['text']):
            try:
                conf = int(float(data['conf'][i]))
            except:
                conf = -1
            if conf < CONF_THRESHOLD or not word.strip():
                continue

            eng_match = re.findall(regex_eng, word)
            tam_match = re.findall(regex_tam, word)
            formula_match = re.findall(regex_formula, word)

            if eng_match and len(word) > 1:
                text_eng.append(''.join(eng_match))
            if tam_match and len(word) > 1:
                text_ta.append(''.join(tam_match))
            if formula_match:
                text_formula.append(''.join(formula_match))

        combined_text = ''
        if text_eng:
            combined_text += ' '.join(text_eng) + '\n'
        if text_ta:
            combined_text += ' '.join(text_ta) + '\n'
        if text_formula:
            combined_text += ' '.join(text_formula)

        st.text_area('Extracted Text', combined_text, height=300)
        if combined_text.strip() and mode != 'Silent Mode':
            tts_lang = 'ta' if language == 'Tamil' else 'en'
            #speak_text = combined_text
            #tts_lang = 'en'

            #if language == 'Tamil':
                #speak_text = Translator().translate(combined_text, dest='ta').text
                #tts_lang = 'ta'
            tts = gTTS(text=combined_text, lang=tts_lang)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            st.audio(audio_bytes.getvalue(), format='audio/mp3')

    # ============================================================
    # ============== YOLO + BLIP PIPELINE (UNCHANGED) =============
    # ============================================================
    if not IS_TEXT_IMAGE:
        custom_model = YOLO("best.pt")
        #custom_model = YOLO(r"D:/Final_Year_Project/trained_model/runs/detect/train3/weights/best.pt")
        #D:\Final_Year_Project\trained_model\runs\detect\train3\weights\best.pt
        #D:\Final_Year_Project\best.pt
        pretrained_model = YOLO("yolov8n.pt")

        custom_results = custom_model(image, conf=0.25)
        pretrained_results = pretrained_model(image, conf=0.25)

        def extract_detections(results, model, offset=0):
            boxes = results[0].boxes
            detections = []
            if boxes is None:
                return detections
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0]) + offset
                label = f"{model.names[int(box.cls[0])]}: {conf:.2f}"
                detections.append((x1, y1, x2, y2, conf, label))
            return detections

        custom_dets = extract_detections(custom_results, custom_model, offset=0)
        pretrained_dets = extract_detections(pretrained_results, pretrained_model, offset=100)
        all_detections = custom_dets + pretrained_dets

        detected_objects = [det[5].split(':')[0] for det in all_detections]
        detected_objects = list(set(detected_objects))
        scene_prompt = 'a scene containing: ' + ', '.join(detected_objects)
        st.write('Scene Prompt:', scene_prompt)

        img = cv2.imread(image_path)
        for det in all_detections:
            x1, y1, x2, y2, conf, label = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption='YOLO Detection + BLIP Ready Image', use_container_width=True)

        processor = AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

        pil_image = Image.fromarray(img_rgb)
        inputs = processor(pil_image, return_tensors='pt')
        out = blip_model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.write('Base Caption:', caption)

        def fuse_caption_natural(caption, detected_objects):
            base_lower = caption.lower()
            new_objects = []
            for obj in detected_objects:
                if obj not in base_lower:
                    new_objects.append(obj)
            if not new_objects:
                return caption
            if len(new_objects) == 1:
                obj_phrase = new_objects[0]
            else:
                obj_phrase = ', '.join(new_objects[:-1]) + ' and ' + new_objects[-1]
            return f"{caption} with {obj_phrase} in the scene"

        final_caption = fuse_caption_natural(caption, detected_objects)
        st.write('Final Caption:', final_caption)

        object_captions = []
        for det in all_detections:
            x1, y1, x2, y2, conf, label = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            inputs = processor(pil_crop, return_tensors='pt')
            out = blip_model.generate(**inputs, max_length=30)
            obj_caption = processor.decode(out[0], skip_special_tokens=True)
            object_captions.append((label, obj_caption))

        for lbl, cap in object_captions:
            st.write(f'{lbl} -> {cap}')

        def combine_object_captions(object_captions):
            phrases = [cap for _, cap in object_captions]
            return '. '.join(phrases)

        object_caption_text = combine_object_captions(object_captions)

        def score_caption(caption, detected_objects):
            caption_lower = caption.lower()
            coverage = sum([1 for obj in detected_objects if obj in caption_lower])
            coverage_score = coverage / len(detected_objects) if detected_objects else 0
            length = len(caption.split())
            if length < 5:
                length_score = 0.3
            elif length > 25:
                length_score = 0.5
            else:
                length_score = 1.0
            return 0.7 * coverage_score + 0.3 * length_score

        candidates = {'base': caption, 'fused': final_caption, 'object': object_caption_text}
        scores = {k: score_caption(v, detected_objects) for k, v in candidates.items()}
        best_type = max(scores, key=scores.get)
        best_caption = candidates[best_type]

        st.subheader('Caption Scores')
        st.json(scores)
        st.success(f'BEST CAPTION TYPE = {best_type}')
        st.write(best_caption)
        if best_caption.strip() and mode != 'Silent Mode':
            #speak_text = best_caption
            tts_lang = 'ta' if language == 'Tamil' else 'en'

            #tts_lang = 'en'
            if language == 'Tamil':
                tamil_caption = GoogleTranslator(source='auto', target='ta').translate(best_caption)
                tts = gTTS(text=tamil_caption, lang=tts_lang)
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                st.audio(audio_bytes.getvalue(), format='audio/mp3')
            else:
        
                tts = gTTS(text=best_caption, lang=tts_lang)
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                st.audio(audio_bytes.getvalue(), format='audio/mp3')


    pass
#python -m streamlit run app.py
