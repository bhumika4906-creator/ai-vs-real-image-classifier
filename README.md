\# AI vs Real Image Classification System



A deep learning pipeline that classifies images as either synthetic AI-generated artwork or authentic photographs using transfer learning with InceptionResNetV2.



Live Web Application: https://ai-vs-real-image-classifier-fgrcnfdq3as7rzybpz7yym.streamlit.app/



\---



\## Overview



The system processes input images through a fine-tuned InceptionResNetV2 feature extractor paired with a custom classification head. Because the trained model weights exceed standard repository file limits, the weights are decoupled from the codebase, hosted on Hugging Face Hub (`bhumika49/ai-vs-real-detector`), and downloaded on demand at runtime.



The user interface is built with Streamlit, providing real-time inference, category outputs, and confidence scores.



\---



\## System Architecture



\- Base Model: InceptionResNetV2 (pre-trained on ImageNet)

\- Input Resolution: 224 x 224 x 3

\- Custom Head: GlobalAveragePooling2D, Dense layers, Dropout, Sigmoid output

\- Model Weight Distribution: Hugging Face Hub API (`att\_model.keras`)

\- Web Interface: Streamlit Community Cloud



\---



\## Tech Stack



\- Python

\- TensorFlow / Keras

\- Streamlit

\- NumPy

\- Pillow (PIL)

\- Hugging Face Hub API







