# # === /prod/app.py ===
# import streamlit as st
# import torch
# import numpy as np
# import cv2
# from PIL import Image
# from utils import cargar_modelo, segmentar_frutas, clasificar_imagen, mostrar_recetas
# import os
# import io
# import base64


# # Estilos CSS personalizados para Streamlit
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
#         color: #eee;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     }

#     .main-container {
#         background: rgba(255, 255, 255, 0.12);
#         border-radius: 16px;
#         padding: 25px 40px 40px 40px;
#         box-shadow: 0 8px 32px 0 rgba(45, 60, 80, 0.37);
#         margin-bottom: 40px;
#     }

#     .title {
#         font-size: 2.8rem;
#         font-weight: 900;
#         color: #dcd6f7;
#         text-align: center;
#         margin-bottom: 10px;
#         text-shadow: 2px 2px 8px #4b3f72;
#     }

#     .description {
#         font-size: 1.25rem;
#         color: #d1c4e9;
#         text-align: center;
#         margin-bottom: 30px;
#     }

#     div.stButton > button {
#         background: #7e57c2;
#         color: white;
#         font-weight: bold;
#         border-radius: 10px;
#         padding: 10px 25px;
#         transition: background-color 0.3s ease;
#         border: none;
#         box-shadow: 0 4px 15px #b39ddb;
#     }
#     div.stButton > button:hover {
#         background: #512da8;
#         box-shadow: 0 6px 20px #311b92;
#     }

#     .caption {
#         font-weight: 600;
#         color: #c5cae9;
#         margin-top: 12px;
#         margin-bottom: 30px;
#         text-align: center;
#         font-size: 1.1rem;
#         text-shadow: 1px 1px 3px #3f3a60;
#     }

#     .stWarning {
#         background-color: #f3e5f5 !important;
#         border-left: 5px solid #7e57c2 !important;
#         color: #4a148c !important;
#         padding: 10px !important;
#         font-weight: bold !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Configurar dispositivo
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Cargar modelo y transformaciones
# MODEL_PATH = os.path.join("prod", "best_efficientnet_b3.pth")
# # MODEL_PATH = "best_efficientnet_b3.pth"
# #MODEL_PATH = os.path.join(os.path.dirname(_file_), "best_efficientnet_b3.pth")
# model, transform, class_names = cargar_modelo(MODEL_PATH, device)

# st.markdown('<div class="main-container">', unsafe_allow_html=True)

# # Mostrar imagen de portada correctamente
# #st.image("portada.jpg", use_container_width=True)
# st.image("prod/portada.jpg", use_container_width=True)


# st.markdown('<p class="description">📸 Subí una imagen de frutas y nuestro modelo te dirá qué frutas ve.</p>', unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Subí una imagen de frutas o arrastrá una imagen", type=["jpg", "jpeg", "png"])
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)
#     st.image(image_np, caption="📷 Imagen original", use_container_width=True)

#     with st.spinner("🔍 Segmentando frutas..."):
#         boxes = segmentar_frutas(image_np, device)

#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     for (x1, y1, x2, y2) in boxes:
#         cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     #st.image(image_rgb, caption="Frutas segmentadas")

#     with st.spinner(" Clasificando frutas..."):
#         resultados = clasificar_imagen(image_np, boxes, model, transform, class_names, device)

#     if resultados:
#         # Convertir imagen original a BGR para dibujar
#         image_final = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
#         # Tamaño original
#         orig_h, orig_w = image_final.shape[:2]

#         # Parámetros para redimensionar la imagen que mostrarás
#         max_width = 800
#         scale_factor = 1.0
#         if orig_w > max_width:
#             scale_factor = max_width / orig_w
#             new_w = max_width
#             new_h = int(orig_h * scale_factor)
#         else:
#             new_w = orig_w
#             new_h = orig_h

#         # Dibujar cajas escalando coordenadas

#         for d in resultados:
#             x, y, w_box, h_box = d["bbox"]
#             label = d["label"]
#             conf = d["conf"]

#             # Escalar coordenadas
#             x_s = int(x * scale_factor)
#             y_s = int(y * scale_factor)
#             w_s = int(w_box * scale_factor)
#             h_s = int(h_box * scale_factor)

#             # Para dibujar en la imagen original primero escalamos la imagen para dibujar después
#             # O podemos dibujar en una copia redimensionada:
#             # Mejor: dibujar después de redimensionar la imagen original

#         # Redimensionar imagen original para dibujar las cajas sobre ella
#         image_final_resized = cv2.resize(image_final, (new_w, new_h))

#         # Dibujar las cajas y texto sobre la imagen redimensionada
#         for d in resultados:
#             x, y, w_box, h_box = d["bbox"]
#             label = d["label"]
#             conf = d["conf"]

#             x_s = int(x * scale_factor)
#             y_s = int(y * scale_factor)
#             w_s = int(w_box * scale_factor)
#             h_s = int(h_box * scale_factor)

#             cv2.rectangle(image_final_resized, (x_s, y_s), (x_s + w_s, y_s + h_s), (0, 255, 0), 2)  # verde
#             # Dentro del bucle que recorre los resultados:
#             text = f"{label} ({conf:.2f})"

#             # Obtener tamaño del texto
#             (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

#             # Calcular punto superior izquierdo para centrar dentro del bbox
#             text_x = x_s + (w_s - text_width) // 2
#             text_y = y_s + (h_s + text_height) // 2  # verticalmente centrado
#             # Dibujar el texto centrado
#             cv2.putText(image_final_resized, text, (text_x, text_y),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
#         # Mostrar la imagen final en RGB
#         st.image(cv2.cvtColor(image_final_resized, cv2.COLOR_BGR2RGB), caption="Clasificación final")
        
#         # Obtener la fruta más confiable
#         fruta_principal = resultados[0]["label"]

#         # Mostrar recetas relacionadas
#         with st.spinner(f"🍰 Buscando recetas con {fruta_principal}..."):
#             imagenes_recetas, titulos_recetas = mostrar_recetas(fruta_principal)

#         if imagenes_recetas:
#             st.subheader(f"🍴 Recetas recomendadas con {fruta_principal}")
#             cols = st.columns(len(imagenes_recetas))
#             for i, col in enumerate(cols):
#                 with col:
#                     st.image(imagenes_recetas[i], caption=titulos_recetas[i], use_container_width=True)
        
#         else:
#             st.info(f"ℹ️ No hay recetas disponibles con {fruta_principal}.")
#     else:
#         st.warning("⚠️ No se encontraron frutas confiables para clasificar.")

# st.markdown('</div>', unsafe_allow_html=True)


# === /prod/app.py ===
# import streamlit as st
# import torch
# import numpy as np
# from PIL import Image
# from utils import cargar_modelo, segmentar_frutas, clasificar_imagen, generar_y_mostrar_receta
# import os
# from dotenv import load_dotenv
# import openai


# # Cargar token desde .env
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Estilos CSS personalizados para Streamlit
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
#         color: #eee;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     }

#     .main-container {
#         background: rgba(255, 255, 255, 0.12);
#         border-radius: 16px;
#         padding: 25px 40px 40px 40px;
#         box-shadow: 0 8px 32px 0 rgba(45, 60, 80, 0.37);
#         margin-bottom: 40px;
#     }

#     .title {
#         font-size: 2.8rem;
#         font-weight: 900;
#         color: #dcd6f7;
#         text-align: center;
#         margin-bottom: 10px;
#         text-shadow: 2px 2px 8px #4b3f72;
#     }

#     .description {
#         font-size: 1.25rem;
#         color: #d1c4e9;
#         text-align: center;
#         margin-bottom: 30px;
#     }

#     div.stButton > button {
#         background: #7e57c2;
#         color: white;
#         font-weight: bold;
#         border-radius: 10px;
#         padding: 10px 25px;
#         transition: background-color 0.3s ease;
#         border: none;
#         box-shadow: 0 4px 15px #b39ddb;
#     }
#     div.stButton > button:hover {
#         background: #512da8;
#         box-shadow: 0 6px 20px #311b92;
#     }

#     .caption {
#         font-weight: 600;
#         color: #c5cae9;
#         margin-top: 12px;
#         margin-bottom: 30px;
#         text-align: center;
#         font-size: 1.1rem;
#         text-shadow: 1px 1px 3px #3f3a60;
#     }

#     .stWarning {
#         background-color: #f3e5f5 !important;
#         border-left: 5px solid #7e57c2 !important;
#         color: #4a148c !important;
#         padding: 10px !important;
#         font-weight: bold !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Configurar dispositivo
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Cargar modelo y transformaciones
# MODEL_PATH = os.path.join("prod", "best_efficientnet_b3.pth")
# model, transform, class_names = cargar_modelo(MODEL_PATH, device)

# st.markdown('<div class="main-container">', unsafe_allow_html=True)

# st.image("prod/portada.jpg", use_container_width=True)

# st.markdown('<p class="description">📸 Subí una imagen de frutas y nuestro modelo te dirá qué frutas ve.</p>', unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Subí una imagen de frutas o arrastrá una imagen", type=["jpg", "jpeg", "png"])
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)
#     st.image(image_np, caption="📷 Imagen original", use_container_width=True)

#     with st.spinner("🔍 Segmentando frutas..."):
#         boxes = segmentar_frutas(image_np, device)

#     with st.spinner(" Clasificando frutas..."):
#         resultados = clasificar_imagen(image_np, boxes, model, transform, class_names, device)

#     if resultados:
#         frutas_detectadas = [r["label"] for r in resultados]
#         frutas_str = ", ".join(frutas_detectadas)

#         st.subheader(f"🧪 Frutas detectadas: {frutas_str}")

#         prompt = f"Generá una receta creativa que use estas frutas: {frutas_str}. Mostrá los ingredientes y el paso a paso."        
#         with st.spinner("🍳 Generando receta con OpenAI..."):
#             try:
#                 respuesta = openai.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=[
#                         {"role": "system", "content": "Sos un chef experto en frutas tropicales."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.7,
#                     max_tokens=400
#                 )
#                 receta = respuesta.choices[0].message.content
#                 st.markdown("### 🍽 Receta generada")
#                 st.markdown(f"""
#     <div style='
#         background-color: #ffffffcc;
#         color: #2e2e2e;
#         padding: 10px 15px;
#         border-radius: 6px;
#         box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
#         font-family: "Segoe UI", sans-serif;
#         line-height: 1.25;
#         font-size: 14px;
#         max-width: 500px;
#         margin: 0 auto;
#     '>
#         <style>
#             ul, ol {{
#                 padding-left: 18px;
#                 margin-top: 4px;
#                 margin-bottom: 4px;
#             }}
#             li {{
#                 margin-bottom: 3px;
#             }}
#         </style>
#         <div style='white-space: pre-wrap; margin: 0; padding: 0;'>{receta}</div>
#     </div>
# """, unsafe_allow_html=True)



#             except Exception as e:
#                 st.warning("⚠️ Error al generar receta con OpenAI")
#                 st.text(str(e))
#     else:
#         st.warning("⚠️ No se encontraron frutas confiables para clasificar.")

# st.markdown('</div>', unsafe_allow_html=True)





# === /prod/app.py ===
# import streamlit as st
# import torch
# import numpy as np
# from PIL import Image
# from utils import cargar_modelo, segmentar_frutas, clasificar_imagen, generar_y_mostrar_receta 
# import os
# from dotenv import load_dotenv
# import openai

# # Cargar token desde .env
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Estilos CSS personalizados para Streamlit
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
#         color: #eee;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     }

#     .main-container {
#         background: rgba(255, 255, 255, 0.12);
#         border-radius: 16px;
#         padding: 25px 40px 40px 40px;
#         box-shadow: 0 8px 32px 0 rgba(45, 60, 80, 0.37);
#         margin-bottom: 40px;
#     }

#     .description {
#         font-size: 1.25rem;
#         color: #d1c4e9;
#         text-align: center;
#         margin-bottom: 30px;
#     }

#     div.stButton > button {
#         background: #7e57c2;
#         color: white;
#         font-weight: bold;
#         border-radius: 10px;
#         padding: 10px 25px;
#         transition: background-color 0.3s ease;
#         border: none;
#         box-shadow: 0 4px 15px #b39ddb;
#     }
#     div.stButton > button:hover {
#         background: #512da8;
#         box-shadow: 0 6px 20px #311b92;
#     }

#     .stWarning {
#         background-color: #f3e5f5 !important;
#         border-left: 5px solid #7e57c2 !important;
#         color: #4a148c !important;
#         padding: 10px !important;
#         font-weight: bold !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Configurar dispositivo
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Cargar modelo
# MODEL_PATH = os.path.join("prod", "best_efficientnet_b3.pth")
# model, transform, class_names = cargar_modelo(MODEL_PATH, device)

# st.markdown('<div class="main-container">', unsafe_allow_html=True)

# st.image("prod/portada.jpg", use_container_width=True)

# st.markdown('<p class="description">📸 Subí una imagen de frutas y nuestro modelo te dirá qué frutas ve.</p>', unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Subí una imagen de frutas o arrastrá una imagen", type=["jpg", "jpeg", "png"])
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)
#     st.image(image_np, caption="📷 Imagen original", use_container_width=True)

#     with st.spinner("🔍 Segmentando frutas..."):
#         boxes = segmentar_frutas(image_np, device)

#     with st.spinner(" Clasificando frutas..."):
#         resultados = clasificar_imagen(image_np, boxes, model, transform, class_names, device)

#     if resultados:
#         frutas_detectadas = [r["label"] for r in resultados]
#         frutas_str = ", ".join(frutas_detectadas)

#         st.subheader(f"🧪 Frutas detectadas: {frutas_str}")

#         # 💡 Llamada única y limpia
#         generar_y_mostrar_receta(frutas_str)
#     else:
#         st.warning("⚠️ No se encontraron frutas confiables para clasificar.")

# st.markdown('</div>', unsafe_allow_html=True)



import streamlit as st
import torch
import numpy as np
from PIL import Image
from utils import cargar_modelo, segmentar_frutas, clasificar_imagen, generar_y_mostrar_receta
import os
from dotenv import load_dotenv
import openai

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

    with st.spinner("🧠 Clasificando frutas..."):
        resultados = clasificar_imagen(frutas_crop, model, transform, class_names, device)

    if resultados:
        frutas_detectadas = [r["label"] for r in resultados]
        frutas_str = ", ".join(frutas_detectadas)
        st.subheader(f"🧪 Frutas detectadas: {frutas_str}")
        generar_y_mostrar_receta(frutas_str)
    else:
        st.warning("⚠️ No se encontraron frutas confiables para clasificar.")

st.markdown('</div>', unsafe_allow_html=True)



