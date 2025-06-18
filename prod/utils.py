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
from huggingface_hub import hf_hub_download




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


def segmentar_frutas(image_np, device):
    #sam_checkpoint = "sam_vit_h_4b8939.pth"
    #sam_checkpoint = os.path.join("prod", "sam_vit_h_4b8939.pth")
    # URL del checkpoint SAM en Hugging Face
    #sam_checkpoint_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth"
    
    # Cargar checkpoint directamente desde la URL
    #sam_checkpoint = torch.hub.load_state_dict_from_url(sam_checkpoint_url, map_location=device)
    # Cargar el checkpoint desde Hugging Face
    sam_checkpoint = hf_hub_download(repo_id="Roccola/sam-vit-h-checkpoint", filename="sam_vit_h_4b8939.pth")

    # Modelo SAM   
    
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

    boxes = []
    scores = []

    for mask in masks:
        seg = mask["segmentation"].astype(np.uint8)
        x, y, w, h = cv2.boundingRect(seg)
        boxes.append([x, y, x + w, y + h])
        scores.append(mask["predicted_iou"])

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
    
    masks = [masks[i] for i in indices]
    filtered_masks = [mask for mask in masks if mask["predicted_iou"] >=confidence_threshold]
    print(f" Después del filtro quedan {len(filtered_masks)} máscaras.")
    
    # Filtramos por confianza
    confidence_threshold = 0.9
    final_boxes = []
    for i in indices:
        if scores[i] >= confidence_threshold:
            x1, y1, x2, y2 = boxes[i]
            final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    return final_boxes


def clasificar_imagen(image_np, boxes, model, transform, class_names, device):
    detecciones = []

    for (x, y, w, h) in boxes:
        pad = int(0.1 * max(w, h))
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, image_np.shape[1])
        y2 = min(y + h + pad, image_np.shape[0])

        crop = image_np[y1:y2, x1:x2]
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
            "bbox": (x1, y1, x2 - x1, y2 - y1)
        })

    # --- Filtrar: una detección por clase, la más confiable ---
    filtradas = []
    if len(detecciones)>0:
        agrupadas = defaultdict(list)
        for d in detecciones:
            agrupadas[d["label"]].append(d)

        for grupo in agrupadas.values():
            mejor = max(grupo, key=lambda d: d["conf"])
            if mejor["conf"] > 0.45:
                filtradas.append(mejor)

    return filtradas



# --- Mostrar recetas e imágenes relacionadas ---
import os
from PIL import Image
import matplotlib.pyplot as plt

def mostrar_recetas(fruta_detectada, carpeta_recetas="../data/Recetario"):
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
