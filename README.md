# AI vs Real Image Classification System

A deep learning pipeline that classifies images as either synthetic AI-generated artwork or authentic photographs using transfer learning with **InceptionResNetV2**.

**Live Demo:** [Open Streamlit Application](https://ai-vs-real-image-classifier-fgrcnfdq3as7rzybpz7yym.streamlit.app/)

---

## Overview

The system processes input images through a fine-tuned InceptionResNetV2 feature extractor paired with a custom classification head. 

Because the trained model weights exceed standard repository file limits, the weights are decoupled from the codebase, hosted on Hugging Face Hub (`bhumika49/ai-vs-real-detector`), and downloaded on demand at runtime.

The user interface is built with Streamlit, providing real-time inference, category predictions, and confidence scoring.

---

## Architecture & Specifications

| Parameter | Specification |
| :--- | :--- |
| **Base Architecture** | InceptionResNetV2 (Pre-trained on ImageNet) |
| **Input Resolution** | 224 × 224 × 3 |
| **Classification Head** | GlobalAveragePooling2D, Dense, Dropout, Sigmoid |
| **Weight Distribution** | Hugging Face Hub API (`att_model.keras`) |
| **Deployment Target** | Streamlit Community Cloud |

---

## Tech Stack

- **Framework:** TensorFlow / Keras
- **Web App:** Streamlit
- **Model Distribution:** Hugging Face Hub
- **Data & Image Processing:** NumPy, Pillow (PIL)

