import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3
import torch.nn as nn
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from collections import defaultdict
import os
import requests
import streamlit as st
import re
import openai


def cargar_modelo(path_modelo, device):
    class_names = ['Anana', 'Banana', 'Coco', 'Frutilla', 'Higo',
                   'Manzana', 'Mora', 'Naranja', 'Palta', 'Pera']

    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )
    model.load_state_dict(torch.load(path_modelo, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return model, transform, class_names

#Revisa si ya está descargado el archivo del modelo SAM y devuelve la ruta local, sino esta descargado, lo descarga y lo coloca en prod
#⚠️ Esto es importante porque sin ese archivo, el modelo no puede segmentar nada.
def obtener_checkpoint_sam():
    local_path = os.path.join("prod", "sam_vit_h_4b8939.pth")
    if not os.path.exists(local_path):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        print("⏳ Descargando checkpoint de SAM...")
        r = requests.get(url, stream=True)
        os.makedirs("prod", exist_ok=True)
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Checkpoint descargado en:", local_path)
    return local_path

#Calcula el IOU (Intersection over Union) entre dos cajas delimitadoras.
#El IOU mide cuánto se solapan dos regiones.
#Se usa para decidir si dos máscaras son de la misma fruta.
def calcular_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

#Toma una lista de máscaras con sus bounding boxes.
#Agrupa máscaras que se solapan bastante (IOU mayor a 0.6).

#Así evitás duplicar frutas si fueron segmentadas en partes separadas.
def agrupar_por_iou(masks, iou_thresh=0.6):
    grupos = []
    usados = set()

    for i, m1 in enumerate(masks):
        if i in usados:
            continue
        grupo = [i]
        box1 = m1["bbox"]
        for j in range(i + 1, len(masks)):
            if j in usados:
                continue
            box2 = masks[j]["bbox"]

            #Recibe dos cajas (bounding boxes).
            #Calcula cuánto se solapan (la intersección).
            #Divide esa intersección por el área total combinada de ambas cajas (la unión).
            #El resultado es un número entre 0 y 1:
            #Cerca de 1: las cajas son muy similares o se superponen mucho.
            #Cerca de 0: están muy separadas.
            iou = calcular_iou(box1, box2)
            if iou >= iou_thresh:
                grupo.append(j)
                usados.add(j)
        usados.add(i)
        grupos.append(grupo)

    return grupos

#Este es el corazón de la segmentación.
def segmentar_frutas(image_np, device):
    #Llama a obtener_checkpoint_sam() para obtener el modelo SAM.
    sam_checkpoint = obtener_checkpoint_sam()
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device).eval()
    #Carga y configura el modelo SAM para segmentar objetos automáticamente
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=12,
        pred_iou_thresh=0.95,
        stability_score_thresh=0.95,
        min_mask_region_area=15000,
        crop_n_layers=0
    )
    #Filtra máscaras con poca confianza (menor a 0.9).
    confidence_threshold = 0.9

    #Aplica el modelo sobre la imagen:
    masks = mask_generator.generate(image_np)
    

    #Para cada máscara válida:
    #Convierte a una máscara binaria.
    #Calcula su bounding box y la guarda.
    masks_filtradas = []
    for m in masks:
        if m["predicted_iou"] >= confidence_threshold:
            seg = m["segmentation"].astype(np.uint8)
            x, y, w, h = cv2.boundingRect(seg)
            m["bbox"] = [x, y, x + w, y + h]
            masks_filtradas.append(m)

    #Usa agrupar_por_iou() para combinar máscaras superpuestas.
    grupos = agrupar_por_iou(masks_filtradas, iou_thresh=0.6)

    frutas_crop = []

    #Para cada grupo de máscaras:
    #Las fusiona.
    #Calcula la caja alrededor.
    #Recorta la imagen en esa caja.
    #Aplica la máscara para dejar solo la fruta visible.
    for grupo in grupos:
        mask_combinada = np.zeros(image_np.shape[:2], dtype=np.uint8)
        for idx in grupo:
            mask_combinada = cv2.bitwise_or(mask_combinada, masks_filtradas[idx]["segmentation"].astype(np.uint8))
        x, y, w, h = cv2.boundingRect(mask_combinada)
        recorte = image_np[y:y+h, x:x+w].copy()
        mask_crop = mask_combinada[y:y+h, x:x+w]
        fruta_visible = cv2.bitwise_and(recorte, recorte, mask=mask_crop)
        frutas_crop.append(fruta_visible)

    print(f"\033[32mSe generaron {len(frutas_crop)} regiones para clasificar.")
    #Devuelve una lista con cada fruta recortada (frutas_crop).
    return frutas_crop


def generar_y_mostrar_receta(frutas_str: str):
    prompt = f"Generá una receta creativa que use estas frutas: {frutas_str}. Mostrá los ingredientes y el paso a paso."
    st.markdown("### 🍽 Receta generada")

    with st.spinner("🍳 Generando receta con OpenAI..."):
        try:
            respuesta = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sos un chef experto en frutas tropicales."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            receta = respuesta.choices[0].message.content
            render_receta_compacta(receta)
        except Exception as e:
            st.warning("⚠️ Ocurrió un error al generar la receta.")
            st.text(str(e))



def clasificar_frutas(frutas_crop, model, transform, class_names, device, threshold=0.7):
    if not frutas_crop:
        return []

    batch = torch.stack([transform(Image.fromarray(cv2.resize(f, (300, 300)))) for f in frutas_crop]).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

    return [
        {"label": class_names[p.item()], "conf": c.item(), "imagen": frutas_crop[i]}
        for i, (p, c) in enumerate(zip(preds, confs)) if c.item() > threshold
    ]


def render_receta_compacta(receta: str):
    receta = receta.strip()
    receta = re.sub(r'\n{2,}', '\n', receta)
    receta = re.sub(r'[ \t]{2,}', ' ', receta)
    receta = re.sub(r"\*\*([^\n*]+)\*\*", r"<h4 style='margin-top:16px;margin-bottom:6px;'>\1</h4>", receta)
    receta = re.sub(r"\*([^\n*]+)\*", r"<strong>\1</strong>", receta)

    def reemplazar_lista_no_ordenada(match):
        items = match.group(0).strip().split("\n")
        lis = "".join([f"<li>{item[2:].strip()}</li>" for item in items])
        return f"<ul style='margin-top:0;margin-bottom:10px;'>{lis}</ul>"

    receta = re.sub(r"(?:(?:^- .+\n?)+)", reemplazar_lista_no_ordenada, receta, flags=re.MULTILINE)

    def reemplazar_lista_ordenada(match):
        items = match.group(0).strip().split("\n")
        lis = "".join([f"<li>{re.sub(r'^\d+\.\s+', '', item)}</li>" for item in items])
        return f"<ol style='margin-top:0;margin-bottom:10px;padding-left:20px;'>{lis}</ol>"

    receta = re.sub(r"(?:(?:^\d+\. .+\n?)+)", reemplazar_lista_ordenada, receta, flags=re.MULTILINE)

    st.markdown(f"""
        <div style='
            background-color: #ffffffcc;
            color: #2e2e2e;
            padding: 12px 18px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            font-family: \"Segoe UI\", sans-serif;
            line-height: 1.5;
            font-size: 15px;
            max-width: 650px;
            margin: 0 auto;
        '>
            {receta}
        </div>
    """, unsafe_allow_html=True)
