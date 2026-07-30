# 👁️ AI Assistant for the Visually Impaired

A real-time multimodal AI system that helps visually impaired users understand their surroundings — combining **object detection**, **image captioning**, **text recognition (OCR)**, and **voice output** in a single Streamlit app.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧠 Overview

People with visual impairments often struggle to independently understand what's around them — objects, people, signage, or printed text. This project combines four AI models into one real-time pipeline to generate a spoken, natural-language description of the environment from a single camera frame.

## ✨ Features

- **Object Detection** — Identifies objects and their positions using YOLOv8 Nano
- **Scene Captioning** — Generates natural-language descriptions using a BLIP Transformer
- **Text Recognition (OCR)** — Reads printed text (signs, labels, documents) via Tesseract OCR
- **Voice Output** — Converts results to speech using gTTS for hands-free accessibility
- **Simple Web Interface** — Built with Streamlit for easy camera capture and instant feedback

## 🏗️ How It Works

```
Camera Input
     │
     ▼
┌─────────────────────────────────────────────┐
│  1. YOLOv8 Nano      → Detects objects       │
│  2. BLIP Transformer → Captions the scene    │
│  3. Tesseract OCR    → Extracts visible text │
└─────────────────────────────────────────────┘
     │
     ▼
 Combined description → gTTS → Audio output
```

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Object Detection | YOLOv8 Nano (Ultralytics) |
| Image Captioning | BLIP Transformer (Hugging Face) |
| Text Extraction | Tesseract OCR |
| Voice Output | gTTS (Google Text-to-Speech) |
| Computer Vision | OpenCV |
| Web App | Streamlit |
| Language | Python |

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Kaviya-3016/AI_Assistant_blind_people.git
cd AI_Assistant_blind_people

# Install system dependencies (Tesseract OCR)
# See packages.txt for apt-level dependencies

# Install Python dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 📁 Project Structure

```
AI_Assistant_blind_people/
├── .streamlit/          # Streamlit configuration
├── app.py               # Main application
├── best.pt               # Trained/fine-tuned model weights
├── yolov8n.pt            # YOLOv8 Nano base weights
├── icon.png               # App icon
├── packages.txt           # System-level dependencies (for deployment)
├── requirements.txt        # Python dependencies
└── README.md
```

## 🎯 Use Case

Point your device's camera at your surroundings, and the app will:
1. Detect and describe objects nearby
2. Read out any visible text
3. Speak a combined, natural-language summary of the scene

Built to give visually impaired users faster, more independent situational awareness.

## 🙋‍♀️ Author

**Kaviya M**
[GitHub](https://github.com/Kaviya-3016) · [LinkedIn](https://linkedin.com/in/kaviya-murugan)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
