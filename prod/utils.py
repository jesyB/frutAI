# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from torchvision.models import efficientnet_b3
# import torch.nn as nn
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# from collections import defaultdict
# import torchvision.ops as ops
# import matplotlib.pyplot as plt
# import os
# from huggingface_hub import hf_hub_download




# masks = []
# final_boxes = []
# detecciones = []
# filtradas = []

# def cargar_modelo(path_modelo, device):
#     class_names = ['Anana', 'Banana', 'Coco', 'Frutilla', 'Higo',
#                    'Manzana', 'Mora', 'Naranja', 'Palta', 'Pera']

#     model = efficientnet_b3(weights=None)
#     model.classifier[1] = nn.Sequential(
#         nn.Linear(model.classifier[1].in_features, 512),
#         nn.ReLU(),
#         nn.Dropout(0.3),
#         nn.Linear(512, len(class_names))
#     )
#     model.load_state_dict(torch.load(path_modelo, map_location=device))
#     model.to(device).eval()

#     transform = transforms.Compose([
#         transforms.Resize((300, 300)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     return model, transform, class_names


# def segmentar_frutas(image_np, device):
#     #sam_checkpoint = "sam_vit_h_4b8939.pth"
#     sam_checkpoint = os.path.join("prod", "sam_vit_h_4b8939.pth")
#     # URL del checkpoint SAM en Hugging Face
#     #sam_checkpoint_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth"
    
#     # Cargar checkpoint directamente desde la URL
#     #sam_checkpoint = torch.hub.load_state_dict_from_url(sam_checkpoint_url, map_location=device)
#     # Cargar el checkpoint desde Hugging Face
#     #sam_checkpoint = hf_hub_download(repo_id="Roccola/sam_vit_h_checkpoint", filename="sam_vit_h_4b8939.pth")

#     # Modelo SAM   
    
#     model_type = "vit_h"

#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device).eval()

#     mask_generator = SamAutomaticMaskGenerator(
#         sam,
#         points_per_side=12,
#         pred_iou_thresh=0.95,
#         stability_score_thresh=0.95,
#         min_mask_region_area=10000,
#         crop_n_layers=0
#     )
#     confidence_threshold = 0.9
#     masks = mask_generator.generate(image_np)

#     boxes = []
#     scores = []

#     for mask in masks:
#         seg = mask["segmentation"].astype(np.uint8)
#         x, y, w, h = cv2.boundingRect(seg)
#         boxes.append([x, y, x + w, y + h])
#         scores.append(mask["predicted_iou"])

#     boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
#     scores_tensor = torch.tensor(scores, dtype=torch.float32)
#     indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
    
#     masks = [masks[i] for i in indices]
#     filtered_masks = [mask for mask in masks if mask["predicted_iou"] >=confidence_threshold]
#     print(f" Después del filtro quedan {len(filtered_masks)} máscaras.")
    
#     # Filtramos por confianza
#     confidence_threshold = 0.9
#     final_boxes = []
#     for i in indices:
#         if scores[i] >= confidence_threshold:
#             x1, y1, x2, y2 = boxes[i]
#             final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

#     return final_boxes


# def clasificar_imagen(image_np, boxes, model, transform, class_names, device):
#     detecciones = []

#     for (x, y, w, h) in boxes:
#         pad = int(0.1 * max(w, h))
#         x1 = max(x - pad, 0)
#         y1 = max(y - pad, 0)
#         x2 = min(x + w + pad, image_np.shape[1])
#         y2 = min(y + h + pad, image_np.shape[0])

#         crop = image_np[y1:y2, x1:x2]
#         if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
#             continue

#         crop_resized = cv2.resize(crop, (300, 300))
#         pil_crop = Image.fromarray(crop_resized)
#         tensor = transform(pil_crop).unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = model(tensor)
#             pred = output.argmax(1).item()
#             conf = torch.softmax(output, dim=1)[0, pred].item()

#         detecciones.append({
#             "label": class_names[pred],
#             "conf": conf,
#             "bbox": (x1, y1, x2 - x1, y2 - y1)
#         })

#     # --- Filtrar: una detección por clase, la más confiable ---
#     filtradas = []
#     if len(detecciones)>0:
#         agrupadas = defaultdict(list)
#         for d in detecciones:
#             agrupadas[d["label"]].append(d)

#         for grupo in agrupadas.values():
#             mejor = max(grupo, key=lambda d: d["conf"])
#             if mejor["conf"] > 0.45:
#                 filtradas.append(mejor)

#     return filtradas



# # --- Mostrar recetas e imágenes relacionadas ---
# import os
# from PIL import Image
# import matplotlib.pyplot as plt

# def mostrar_recetas(fruta_detectada, carpeta_recetas="../data/Recetario"):
#     recetario = {
#         "Anana": ["Helado de ananá natural", "Mousse de ananá", "Tarta tropical de frutas"],
#         "Banana": ["Banana split", "Helado de banana casero", "Tarta fría de banana"],
#         "Coco": ["Helado de coco", "Trufas de coco y chocolate blanco", "Cheesecake de coco"],
#         "Frutilla": ["Helado de frutilla", "Tarta helada de frutilla", "Parfait de frutas"],
#         "Higo": ["Helado de higos con miel", "Higos rellenos con queso crema", "Postre crocante de higos"],
#         "Manzana": ["Manzanas caramelizadas", "Crumble de manzana", "Helado de manzana verde"],
#         "Mora": ["Sorbete de mora", "Copa de moras y crema", "Helado de frutos rojos"],
#         "Naranja": ["Helado de naranja", "Gelatina cítrica con crema", "Mousse de naranja"],
#         "Palta": ["Ensalada de palta con pera", "Palta dulce con yogurt y miel", "Tartitas saladas con palta y frutas"],
#         "Pera": ["Helado de pera", "Tarta dulce de peras con nuez", "Peras al horno con crema chantilly"]
#     }

#     archivo_por_receta = {
#         "Helado de ananá natural": "HeladoAnana.jpg",
#         "Mousse de ananá": "mousseAnana.jpg",
#         "Tarta tropical de frutas": "tortaTropical.jpg",
#         "Banana split": "bananaSplit.jpg",
#         "Helado de banana casero": "heladoBananaCasero.jpg",
#         "Tarta fría de banana": "tartaFriaBanana.jpg",
#         "Helado de coco": "HeladoCoco.jpg",
#         "Trufas de coco y chocolate blanco": "TrufasCoco.jpg",
#         "Cheesecake de coco": "cheesecakeCoco.jpg",
#         "Helado de frutilla": "heladoFrutilla.jpg",
#         "Tarta helada de frutilla": "TartaFrutilla.jpg",
#         "Parfait de frutas": "Parfait.jpg",
#         "Helado de higos con miel": "HeladoHigo.jpg",
#         "Higos rellenos con queso crema": "HigoRelleno.jpg",
#         "Postre crocante de higos": "CrocanteHigo.jpg",
#         "Manzanas caramelizadas": "manzanaCaramelizadas.jpg",
#         "Crumble de manzana": "crumble.jpg",
#         "Helado de manzana verde": "heladoManzanaVerde.jpg",
#         "Sorbete de mora": "sorbeteMora.jpg",
#         "Copa de moras y crema": "MoraYCrema.jpg",
#         "Helado de frutos rojos": "HeladoFrutosRojos.jpg",
#         "Helado de naranja": "heladoNaranja.jpg",
#         "Gelatina cítrica con crema": "GelatinaCrema.jpg",
#         "Mousse de naranja": "MousseNaranja.jpg",
#         "Ensalada de palta con pera": "ensaladaPaltaPera.jpg",
#         "Palta dulce con yogurt y miel": "paltaDulce.jpg",
#         "Tartitas saladas con palta y frutas": "TartaPalta.jpg",
#         "Helado de pera": "HeladoPera.jpg",
#         "Tarta dulce de peras con nuez": "TartaPera.jpg",
#         "Peras al horno con crema chantilly": "PerasAlHorno.jpg"
#     }

#     recetas = recetario.get(fruta_detectada.capitalize(), [])
#     recetas_con_imagen = [r for r in recetas if archivo_por_receta.get(r)]

#     imagenes = []
#     titulos = []
#     for receta in recetas_con_imagen:
#         archivo = archivo_por_receta[receta]
#         path = os.path.join(carpeta_recetas, archivo)
#         try:
#             img = Image.open(path)
#             imagenes.append(img)
#             titulos.append(receta)
#         except Exception as e:
#             print(f"⚠ Error al cargar la receta {receta}: {e}")
    
#     return imagenes, titulos


import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3
import torch.nn as nn
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from collections import defaultdict
import torchvision.ops as ops
import matplotlib.pyplot as plt
import os
import requests

masks = []
final_boxes = []
detecciones = []
filtradas = []

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



from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import cv2
import torch

def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def agrupar_por_iou(masks, iou_thresh=0.5):
    n = len(masks)
    grupos = []
    visitados = [False] * n

    for i in range(n):
        if visitados[i]:
            continue
        grupo = [i]
        visitados[i] = True
        for j in range(i + 1, n):
            if visitados[j]:
                continue
            if calcular_iou(masks[i]["bbox"], masks[j]["bbox"]) > iou_thresh:
                grupo.append(j)
                visitados[j] = True
        grupos.append(grupo)
    return grupos

def segmentar_frutas(image_np, device):
    sam_checkpoint = obtener_checkpoint_sam()
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device).eval()

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=12,
        pred_iou_thresh=0.95,
        stability_score_thresh=0.95,
        min_mask_region_area=10000,
        crop_n_layers=0
    )

    confidence_threshold = 0.9
    masks = mask_generator.generate(image_np)

    # Filtrar y añadir bbox
    masks_filtradas = []
    for m in masks:
        if m["predicted_iou"] >= confidence_threshold:
            seg = m["segmentation"].astype(np.uint8)
            x, y, w, h = cv2.boundingRect(seg)
            m["bbox"] = [x, y, x + w, y + h]
            masks_filtradas.append(m)

    # Agrupar por IoU
    grupos = agrupar_por_iou(masks_filtradas, iou_thresh=0.5)

    # Recortar y guardar imágenes de frutas
    frutas_crop = []
    for grupo in grupos:
        mask_combinada = np.zeros(image_np.shape[:2], dtype=np.uint8)
        for idx in grupo:
            mask_combinada = cv2.bitwise_or(mask_combinada, masks_filtradas[idx]["segmentation"].astype(np.uint8))
        x, y, w, h = cv2.boundingRect(mask_combinada)

        # Aplicar la máscara combinada al recorte
        recorte = image_np[y:y+h, x:x+w].copy()
        mask_crop = mask_combinada[y:y+h, x:x+w]
        fruta_visible = cv2.bitwise_and(recorte, recorte, mask=mask_crop)

        frutas_crop.append(fruta_visible)

    print(f"\033[32mSe generaron {len(frutas_crop)} regiones para clasificar.")
    return frutas_crop


# def segmentar_frutas(image_np, device):
#     sam_checkpoint = obtener_checkpoint_sam()
#     model_type = "vit_h"

#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device).eval()

#     mask_generator = SamAutomaticMaskGenerator(
#         sam,
#         points_per_side=12,
#         pred_iou_thresh=0.95,
#         stability_score_thresh=0.95,
#         min_mask_region_area=10000,
#         crop_n_layers=0
#     )
#     confidence_threshold = 0.9
#     masks = mask_generator.generate(image_np)

#     boxes = []
#     scores = []

#     for mask in masks:
#         seg = mask["segmentation"].astype(np.uint8)
#         x, y, w, h = cv2.boundingRect(seg)
#         boxes.append([x, y, x + w, y + h])
#         scores.append(mask["predicted_iou"])

#     boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
#     scores_tensor = torch.tensor(scores, dtype=torch.float32)
#     indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

#     masks = [masks[i] for i in indices]
#     filtered_masks = [mask for mask in masks if mask["predicted_iou"] >=confidence_threshold]
#     print(f" [33mDespués del filtro quedan {len(filtered_masks)} máscaras.")

#     final_boxes = []
#     for i in indices:
#         if scores[i] >= confidence_threshold:
#             x1, y1, x2, y2 = boxes[i]
#             final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

#     return final_boxes

# def clasificar_imagen(image_np, boxes, model, transform, class_names, device):
#     detecciones = []

#     for (x, y, w, h) in boxes:
#         pad = int(0.1 * max(w, h))
#         x1 = max(x - pad, 0)
#         y1 = max(y - pad, 0)
#         x2 = min(x + w + pad, image_np.shape[1])
#         y2 = min(y + h + pad, image_np.shape[0])

#         crop = image_np[y1:y2, x1:x2]
#         if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
#             continue

#         crop_resized = cv2.resize(crop, (300, 300))
#         pil_crop = Image.fromarray(crop_resized)
#         tensor = transform(pil_crop).unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = model(tensor)
#             pred = output.argmax(1).item()
#             conf = torch.softmax(output, dim=1)[0, pred].item()

#         detecciones.append({
#             "label": class_names[pred],
#             "conf": conf,
#             "bbox": (x1, y1, x2 - x1, y2 - y1)
#         })

#     filtradas = []
#     if len(detecciones) > 0:
#         agrupadas = defaultdict(list)
#         for d in detecciones:
#             agrupadas[d["label"]].append(d)

#         for grupo in agrupadas.values():
#             mejor = max(grupo, key=lambda d: d["conf"])
#             if mejor["conf"] > 0.45:
#                 filtradas.append(mejor)

#     return filtradas



# Ahora Recibe lotes de imagenes no boundingBoxes

def clasificar_imagen(frutas_crop, model, transform, class_names, device):
    detecciones = []

    for idx, crop in enumerate(frutas_crop):
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        crop_resized = cv2.resize(crop, (300, 300))
        pil_crop = Image.fromarray(crop_resized)
        tensor = transform(pil_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(1).item()
            conf = torch.softmax(output, dim=1)[0, pred].item()

        detecciones.append({
            "label": class_names[pred],
            "conf": conf,
            "idx": idx  # opcional, por si querés referenciar la fruta original
        })

    # Agrupamiento y filtrado por confianza
    filtradas = []
    if detecciones:
        agrupadas = defaultdict(list)
        for d in detecciones:
            agrupadas[d["label"]].append(d)

        for grupo in agrupadas.values():
            mejor = max(grupo, key=lambda d: d["conf"])
            if mejor["conf"] > 0.45:
                filtradas.append(mejor)

    return filtradas




def mostrar_recetas(fruta_detectada):
    base_dir = os.path.dirname(__file__)
    carpeta_recetas = os.path.join(base_dir, "..", "data", "Recetario")

    recetario = {
        "Anana": ["Helado de ananá natural", "Mousse de ananá", "Tarta tropical de frutas"],
        "Banana": ["Banana split", "Helado de banana casero", "Tarta fría de banana"],
        "Coco": ["Helado de coco", "Trufas de coco y chocolate blanco", "Cheesecake de coco"],
        "Frutilla": ["Helado de frutilla", "Tarta helada de frutilla", "Parfait de frutas"],
        "Higo": ["Helado de higos con miel", "Higos rellenos con queso crema", "Postre crocante de higos"],
        "Manzana": ["Manzanas caramelizadas", "Crumble de manzana", "Helado de manzana verde"],
        "Mora": ["Sorbete de mora", "Copa de moras y crema", "Helado de frutos rojos"],
        "Naranja": ["Helado de naranja", "Gelatina cítrica con crema", "Mousse de naranja"],
        "Palta": ["Ensalada de palta con pera", "Palta dulce con yogurt y miel", "Tartitas saladas con palta y frutas"],
        "Pera": ["Helado de pera", "Tarta dulce de peras con nuez", "Peras al horno con crema chantilly"]
    }

    archivo_por_receta = {
        "Helado de ananá natural": "HeladoAnana.jpg",
        "Mousse de ananá": "mousseAnana.jpg",
        "Tarta tropical de frutas": "tortaTropical.jpg",
        "Banana split": "bananaSplit.jpg",
        "Helado de banana casero": "heladoBananaCasero.jpg",
        "Tarta fría de banana": "tartaFriaBanana.jpg",
        "Helado de coco": "HeladoCoco.jpg",
        "Trufas de coco y chocolate blanco": "TrufasCoco.jpg",
        "Cheesecake de coco": "cheesecakeCoco.jpg",
        "Helado de frutilla": "heladoFrutilla.jpg",
        "Tarta helada de frutilla": "TartaFrutilla.jpg",
        "Parfait de frutas": "Parfait.jpg",
        "Helado de higos con miel": "HeladoHigo.jpg",
        "Higos rellenos con queso crema": "HigoRelleno.jpg",
        "Postre crocante de higos": "CrocanteHigo.jpg",
        "Manzanas caramelizadas": "manzanaCaramelizadas.jpg",
        "Crumble de manzana": "crumble.jpg",
        "Helado de manzana verde": "heladoManzanaVerde.jpg",
        "Sorbete de mora": "sorbeteMora.jpg",
        "Copa de moras y crema": "MoraYCrema.jpg",
        "Helado de frutos rojos": "HeladoFrutosRojos.jpg",
        "Helado de naranja": "heladoNaranja.jpg",
        "Gelatina cítrica con crema": "GelatinaCrema.jpg",
        "Mousse de naranja": "MousseNaranja.jpg",
        "Ensalada de palta con pera": "ensaladaPaltaPera.jpg",
        "Palta dulce con yogurt y miel": "paltaDulce.jpg",
        "Tartitas saladas con palta y frutas": "TartaPalta.jpg",
        "Helado de pera": "HeladoPera.jpg",
        "Tarta dulce de peras con nuez": "TartaPera.jpg",
        "Peras al horno con crema chantilly": "PerasAlHorno.jpg"
    }

    recetas = recetario.get(fruta_detectada.capitalize(), [])
    recetas_con_imagen = [r for r in recetas if archivo_por_receta.get(r)]

    imagenes = []
    titulos = []
    for receta in recetas_con_imagen:
        archivo = archivo_por_receta[receta]
        path = os.path.join(carpeta_recetas, archivo)
        try:
            img = Image.open(path)
            imagenes.append(img)
            titulos.append(receta)
        except Exception as e:
            print(f"⚠ Error al cargar la receta {receta}: {e}")

    return imagenes, titulos




import openai
import streamlit as st
import re


import streamlit as st
import re

def render_receta_compacta(receta: str):
    # Limpieza básica
    receta = receta.strip()
    receta = re.sub(r'\n{2,}', '\n', receta)
    receta = re.sub(r'[ \t]{2,}', ' ', receta)

    # Reemplazo de encabezados comunes por HTML estilizado
    receta = re.sub(r"\*\*([^\n*]+)\*\*", r"<h4 style='margin-top:16px;margin-bottom:6px;'>\1</h4>", receta)
    receta = re.sub(r"\*([^\n*]+)\*", r"<strong>\1</strong>", receta)

    # Convertir listas no ordenadas
    receta = re.sub(r"(?m)^- (.+)", r"<li>\1</li>", receta)
    receta = re.sub(r"(<li>.+?</li>)", r"<ul style='margin-top:0;margin-bottom:10px;'>\1</ul>", receta, flags=re.DOTALL)

    # Convertir listas ordenadas
    receta = re.sub(r"(?m)^\d+\. (.+)", r"<li>\1</li>", receta)
    receta = re.sub(r"(<li>.+?</li>)", r"<ol style='margin-top:0;margin-bottom:10px;padding-left:20px;'>\1</ol>", receta, flags=re.DOTALL)

    # Render final
    st.markdown(f"""
        <div style='
            background-color: #ffffffcc;
            color: #2e2e2e;
            padding: 12px 18px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            font-family: "Segoe UI", sans-serif;
            line-height: 1.5;
            font-size: 15px;
            max-width: 650px;
            margin: 0 auto;
        '>
            {receta}
        </div>
    """, unsafe_allow_html=True)




# def generar_y_mostrar_receta(frutas_str: str):
#     prompt = f"Generá una receta creativa que use estas frutas: {frutas_str}. Mostrá los ingredientes y el paso a paso."
#     st.markdown("### 🍽 Receta generada")

#     with st.spinner("🍳 Generando receta con OpenAI..."):
#         try:
#             respuesta = openai.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": "Sos un chef experto en frutas tropicales."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.7,
#                 max_tokens=400
#             )
#             receta = respuesta.choices[0].message.content
#             render_receta_compacta(receta)
#         except Exception as e:
#             st.warning("⚠️ Error al generar receta con OpenAI")
#             st.text(str(e))




def generar_y_mostrar_receta(frutas_str: str):
    prompt = f"Generá una receta creativa que use estas frutas: {frutas_str}. Mostrá los ingredientes y el paso a paso."
    st.markdown("### 🍽 Receta generada")
    
    with st.spinner("🍳 Generando receta con OpenAI..."):
        try:
            # 1. Obtener la receta en texto
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
            st.warning("⚠️ Ocurrió un error al generar la receta o la imagen.")
            st.text(str(e))






# def mostrar_imagen_receta_mini(frutas_str: str):
#     prompt = (
#         f"Fotografía en vista cenital de una receta realista con {frutas_str}, "
#         f"servida en un plato blanco, fondo claro, luz natural. Sin texto, estilo simple y limpio."
#     )

#     with st.spinner("🖼️ Generando imagen representativa..."):
#         try:
#             respuesta = openai.images.generate(
#                 model="dall-e-3",
#                 prompt=prompt,
#                 n=1,
#                 size="1024x1024",  # el tamaño mínimo para dall-e-3, pero lo mostramos reducido
#                 quality="standard"
#             )
#             url = respuesta.data[0].url
#             st.image(url, width=250, caption="Vista previa de la receta")
#         except Exception as e:
#             st.warning("⚠️ No se pudo generar la imagen de la receta.")
#             st.text(str(e))
