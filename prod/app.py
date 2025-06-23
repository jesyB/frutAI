import streamlit as st
import torch
import numpy as np
from PIL import Image
from utils import cargar_modelo, segmentar_frutas, generar_y_mostrar_receta
import os
from dotenv import load_dotenv
import openai
import cv2

# Cargar token desde .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: #eee;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .main-container {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 25px 40px 40px 40px;
        box-shadow: 0 8px 32px 0 rgba(45, 60, 80, 0.37);
        margin-bottom: 40px;
    }

    .description {
        font-size: 1.25rem;
        color: #d1c4e9;
        text-align: center;
        margin-bottom: 30px;
    }

    div.stButton > button {
        background: #7e57c2;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 25px;
        transition: background-color 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px #b39ddb;
    }
    div.stButton > button:hover {
        background: #512da8;
        box-shadow: 0 6px 20px #311b92;
    }

    .stWarning {
        background-color: #f3e5f5 !important;
        border-left: 5px solid #7e57c2 !important;
        color: #4a148c !important;
        padding: 10px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
MODEL_PATH = os.path.join("prod", "best_efficientnet_b3.pth")
model, transform, class_names = cargar_modelo(MODEL_PATH, device)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Imagen portada
st.image("prod/portada.jpg", use_container_width=True)
st.markdown('<p class="description">📸 Subí una imagen de frutas y nuestro modelo te dirá qué frutas ve.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Subí una imagen de frutas o arrastrá una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="📷 Imagen original", use_container_width=True)

    with st.spinner("🔍 Segmentando frutas..."):
        frutas_crop = segmentar_frutas(image_np, device)

    # Filtrar frutas demasiado pequeñas
    frutas_crop = [f for f in frutas_crop if f.shape[0] >= 50 and f.shape[1] >= 50]

    with st.spinner("🧠 Clasificando frutas..."):
        if frutas_crop:
            batch = torch.stack([transform(Image.fromarray(cv2.resize(f, (300, 300)))) for f in frutas_crop]).to(device)
            with torch.no_grad():
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)

            resultados = [
                {"label": class_names[p.item()], "conf": c.item(), "imagen": frutas_crop[i]}
                for i, (p, c) in enumerate(zip(preds, confs)) if c.item() > 0.7
            ]
        else:
            resultados = []

    if resultados:
        st.subheader("🍓 Frutas detectadas")

        cols = st.columns(min(4, len(resultados)))  # Hasta 4 columnas
        for i, resultado in enumerate(resultados):
            col = cols[i % len(cols)]
            with col:
                st.image(resultado["imagen"], caption=f"{resultado['label']} ({resultado['conf']:.2%})", use_container_width=True)

        frutas_detectadas = [r["label"] for r in resultados]
        frutas_str = ", ".join(frutas_detectadas)
        generar_y_mostrar_receta(frutas_str)
    else:
        st.warning("⚠️ No se encontraron frutas confiables para clasificar.")

st.markdown('</div>', unsafe_allow_html=True)
