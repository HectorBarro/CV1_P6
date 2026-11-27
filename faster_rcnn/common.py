"""common: código compartido por los cuadernos de Faster R-CNN v2 sobre pills-sxdht.

El cuaderno del baseline (`baseline-faster-r-cnn-v2.ipynb`) define todo el pipeline en línea.
Aquí está el mismo código, sin cambios de comportamiento, para que cada cuaderno de mejora
solo contenga lo que cambia respecto al baseline. Así todos los experimentos usan exactamente
la misma limpieza de clases, los mismos DataLoaders y la misma evaluación.

Contenido:
    - Datos: carga de Hugging Face, limpieza de clases, `PillsDataset`, DataLoaders.
    - Modelo: `build_model` (Faster R-CNN v2 COCO_V1 con cabeza nueva).
    - Entrenamiento: `train_one_epoch`, `validation_loss`, `evaluate`, `fit`, schedulers.
    - Evaluación en test: `test_report`, `plot_pr_curves`, `plot_examples`, `measure_inference`, y
      `evaluate_predictions` / `results_report` para predicciones ya calculadas (TTA, ensembles).
    - Fusión de detecciones de varias vistas o modelos: `wbf` (Weighted Boxes Fusion) y `nms_fusion`.
    - Análisis de errores: `collect_predictions`, `tide_analysis` (usa `mapviz.py`) y
      `error_analysis_figures` (AP vs IoU, TIDE, matriz de confusión, peores imágenes).
    - Comparación: `load_experiment`, `compare_metrics`, `compare_tide`, `plot_history_comparison`.

Estructura del repositorio: cada modelo tiene su carpeta (este código está en `faster_rcnn/`) y
`mapviz.py` está en la raíz, compartido. Los cuadernos se ejecutan con la carpeta del modelo como
directorio de trabajo (lo que hace Jupyter por defecto); en Kaggle o Colab los ficheros que faltan
se descargan del repositorio.

Convención de resultados: cada experimento escribe en `faster_rcnn/resultados/<nombre>/`:
    results.json  (métricas y configuración)   history.json  (curvas por época)
    tide.json     (descomposición TIDE)        preds_<split>.json (predicciones, para ensembles/TTA)
    best_model.pth y figuras .png (estos no se suben al repositorio)
"""
import copy
import json
import random
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision import tv_tensors
from torchvision.transforms import v2

REPO_RAW = "https://raw.githubusercontent.com/HectorBarro/CV1_P6/main"
MODEL_DIR = "faster_rcnn"            # carpeta de este modelo dentro del repositorio
RESULTS_ROOT = Path("resultados")    # relativa al directorio de trabajo (= carpeta del modelo)

DATASET = "Francesco/pills-sxdht"
# Limpieza de clases (sección 4 del baseline): se descartan la supercategoría `pills` (id 0,
# sin anotaciones) e `Ibuphil 600 mg` (id 2, una sola instancia en train). Las etiquetas
# se reasignan a 1..7 porque la 0 es el fondo en torchvision.
KEEP_IDS = [1, 3, 4, 5, 6, 7, 8]
DROP_IDS = {0, 2}
ID2LABEL = {orig: new for new, orig in enumerate(KEEP_IDS, start=1)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def describe_device():
    return f"{device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")


def hardware_name():
    return torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"


def ensure_file(local_path, repo_path=None):
    """Devuelve `local_path`; si no existe (Kaggle / Colab), lo descarga del repositorio.

    `repo_path` es su ruta desde la raíz del repositorio. Por defecto, la misma ruta dentro de
    la carpeta del modelo (`faster_rcnn/<local_path>`).
    """
    path = Path(local_path)
    if path.exists():
        return path
    repo_path = repo_path or f"{MODEL_DIR}/{path.as_posix()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{REPO_RAW}/{repo_path}", path)
    except Exception as e:
        raise FileNotFoundError(f"No se encuentra {local_path} ni se pudo descargar {repo_path} del repositorio ({e})")
    return path


def _save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def load_splits():
    """Particiones de Hugging Face y lista de clases del modelo (fondo + 7)."""
    from datasets import load_dataset

    hf = {
        "train": load_dataset(DATASET, split="train"),
        "valid": load_dataset(DATASET, split="validation"),
        "test": load_dataset(DATASET, split="test"),
    }
    all_names = hf["train"].features["objects"]["category"].feature.names
    classes = ["__background__"] + [all_names[i] for i in KEEP_IDS]
    return hf, classes


class PillsDataset(Dataset):
    # Adapta el dataset de Hugging Face al formato de los detectores de torchvision

    def __init__(self, hf_ds, transforms=None):
        # Se descartan las imágenes con clases eliminadas (la pastilla seguiría visible sin anotar)
        self.ds = hf_ds.filter(lambda o: not any(c in DROP_IDS for c in o["category"]),
                               input_columns="objects")
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"].convert("RGB")
        objs = item["objects"]

        # COCO [x, y, w, h] -> [x1, y1, x2, y2]
        boxes = torch.tensor(objs["bbox"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(img.height, img.width)),
            "labels": torch.tensor([ID2LABEL[c] for c in objs["category"]], dtype=torch.int64),
            "image_id": item["image_id"],
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def base_transforms():
    """Transformación del baseline: solo pasa a tensor float en [0, 1]. Sin aumento de datos."""
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def collate_fn(batch):
    return tuple(zip(*batch))


def make_datasets(hf, train_tf=None, eval_tf=None):
    """PillsDataset de las tres particiones. `train_tf` solo se aplica a train."""
    eval_tf = eval_tf or base_transforms()
    train_tf = train_tf or eval_tf
    return {
        "train": PillsDataset(hf["train"], train_tf),
        "valid": PillsDataset(hf["valid"], eval_tf),
        "test": PillsDataset(hf["test"], eval_tf),
    }


def make_loaders(ds, batch_size, seed):
    # Generador con semilla: el orden aleatorio de train es el mismo en cada ejecución.
    # num_workers=0: en Windows + Jupyter los workers no pueden importar clases del cuaderno.
    g = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                            num_workers=0, generator=g),
        "valid": DataLoader(ds["valid"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0),
        "test": DataLoader(ds["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0),
    }


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
def build_model(num_classes, seed=None, **model_kwargs):
    """Faster R-CNN v2 con pesos COCO_V1 y un predictor nuevo de `num_classes` salidas.

    `model_kwargs` se pasa al constructor de torchvision (p. ej. `min_size=640, max_size=640`
    para cambiar la resolución, o `trainable_backbone_layers=5` para descongelar todo).
    """
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if seed is not None:
        set_seed(seed)  # la inicialización de la nueva cabeza también debe ser reproducible
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **model_kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)


def count_parameters(model):
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_total, n_train


# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------
def to_device(images, targets):
    images = [img.to(device) for img in images]
    targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
    return images, targets


def train_one_epoch(model, loader, optimizer, scheduler=None):
    """Una época. Si se pasa `scheduler`, se avanza un paso por iteración (no por época)."""
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images, targets = to_device(images, targets)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total += loss.item() * len(images)
        n += len(images)
    return total / n


@torch.no_grad()
def validation_loss(model, loader):
    # Modo train para obtener las pérdidas, pero con las BatchNorm en eval para no tocar sus estadísticas
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    total, n = 0.0, 0
    for images, targets in loader:
        images, targets = to_device(images, targets)
        loss = sum(model(images, targets).values())
        total += loss.item() * len(images)
        n += len(images)
    model.eval()
    return total / n


@torch.no_grad()
def evaluate(model, loader, extended=False):
    # extended=True añade AP por clase y la matriz de precisión para las curvas PR
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  class_metrics=extended, extended_summary=extended)
    for images, targets in loader:
        images, targets = to_device(images, targets)
        preds = model(images)
        metric.update(
            [{k: v.cpu() for k, v in p.items()} for p in preds],
            [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets],
        )
    return metric.compute()


def evaluate_predictions(preds, targets, extended=False):
    """Como `evaluate`, pero a partir de predicciones ya calculadas (TTA, ensembles...).
    `preds`: lista de dicts con boxes/scores/labels; `targets`: lista de dicts con boxes/labels."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  class_metrics=extended, extended_summary=extended)
    metric.update([{k: v.cpu() for k, v in p.items()} for p in preds],
                  [{"boxes": torch.as_tensor(t["boxes"]).cpu(), "labels": t["labels"].cpu()} for t in targets])
    return metric.compute()


def warmup_cosine_scheduler(optimizer, total_iters, warmup_iters, warmup_factor=0.001, min_factor=0.0):
    """LR por iteración: subida lineal desde `warmup_factor·lr` hasta `lr` en `warmup_iters`
    iteraciones y después bajada en coseno hasta `min_factor·lr` al final del entrenamiento."""
    def factor(it):
        if it < warmup_iters:
            alpha = it / max(1, warmup_iters)
            return warmup_factor * (1 - alpha) + alpha
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return min_factor + (1 - min_factor) * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def fit(model, loaders, optimizer, num_epochs, out_dir, scheduler=None):
    """Bucle de entrenamiento del baseline: registra pérdidas y mAP de validación por época y
    guarda el modelo con mayor mAP@0.5:0.95 en validación. Test no se usa para nada."""
    out_dir = Path(out_dir)
    history = {"train_loss": [], "val_loss": [], "val_map": [], "val_map50": [], "lr": []}
    best_map, best_epoch, best_state = -1.0, 0, None
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        lr_epoch = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, loaders["train"], optimizer, scheduler)
        val_loss = validation_loss(model, loaders["valid"])
        m = evaluate(model, loaders["valid"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(m["map"].item())
        history["val_map50"].append(m["map_50"].item())
        history["lr"].append(lr_epoch)

        if m["map"].item() > best_map:
            best_map, best_epoch = m["map"].item(), epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_dir / "best_model.pth")

        print(f"Época {epoch:2d}/{num_epochs} | lr {lr_epoch:.5f} | loss train {train_loss:.4f} | "
              f"loss val {val_loss:.4f} | val mAP {m['map']:.4f} | val mAP@50 {m['map_50']:.4f} | "
              f"{time.time() - t0:.0f}s")

    train_time = time.time() - t_start
    model.load_state_dict(best_state)
    _save_json(history, out_dir / "history.json")
    print(f"\nMejor época: {best_epoch} (val mAP {best_map:.4f}) | tiempo total: {train_time / 60:.1f} min")
    return dict(history=history, best_epoch=best_epoch, best_map=best_map, train_time=train_time)


# ----------------------------------------------------------------------------
# Evaluación en test
# ----------------------------------------------------------------------------
def _ap_from_precision(p):
    p = p[p > -1]
    return p.mean().item() if p.numel() else float("nan")


def test_report(model, loader, classes, verbose=True):
    """mAP de COCO en test y AP por clase, igual que la sección 10 del baseline."""
    return results_report(evaluate(model, loader, extended=True), classes, verbose)


def results_report(res, classes, verbose=True):
    """AP por clase e impresión de las métricas a partir del resultado de `evaluate(..., extended=True)`
    o de `evaluate_predictions(..., extended=True)`."""
    precision = res["precision"]  # [IoU, recall, clase, área, máx. dets]; -1 = sin objetos
    per_class = {}
    for k, c in enumerate(res["classes"].tolist()):
        per_class[classes[c]] = {
            "AP": _ap_from_precision(precision[:, :, k, 0, -1]),
            "AP50": _ap_from_precision(precision[0, :, k, 0, -1]),
        }
    if verbose:
        print(f"mAP@0.5:0.95 : {res['map']:.4f}")
        print(f"mAP@0.5      : {res['map_50']:.4f}")
        print(f"mAP@0.75     : {res['map_75']:.4f}")
        print(f"mAP pequeño / mediano / grande : {res['map_small']:.4f} / {res['map_medium']:.4f} / "
              f"{res['map_large']:.4f}")
        print(f"mAR@100      : {res['mar_100']:.4f}")
        print(f"\n{'clase':<22}{'AP50:95':>9}{'AP50':>9}")
        for name, v in per_class.items():
            print(f"{name:<22}{v['AP']:>9.4f}{v['AP50']:>9.4f}")
    return res, per_class


def plot_pr_curves(res, classes, out_path=None):
    precision = res["precision"]
    recall_thr = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (t_idx, iou_txt) in zip(axes, [(0, "0.5"), (5, "0.75")]):
        for k, c in enumerate(res["classes"].tolist()):
            p = precision[t_idx, :, k, 0, -1]
            if (p > -1).any():
                ap = _ap_from_precision(p)
                ax.plot(recall_thr, np.clip(p.numpy(), 0, None), label=f"{classes[c]} (AP {ap:.2f})")
        ax.set_title(f"Curvas PR en test (IoU = {iou_txt})")
        ax.set_xlabel("recall")
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("precisión")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
    plt.show()


@torch.no_grad()
def plot_examples(model, dataset, classes, out_path, n=6, seed=42, score_thr=0.5, predict=None):
    """Análisis cualitativo (sección 12 del baseline): imágenes de test al azar con las cajas reales
    (verde) y las predicciones con score >= score_thr (rojo). Con la misma semilla que el baseline
    salen las mismas imágenes, para comparar una a una. `predict(img) -> dict` sustituye a la
    llamada al modelo (p. ej. para TTA)."""
    from torchvision.utils import draw_bounding_boxes

    predict = predict or (lambda img: model([img.to(device)])[0])
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    if model is not None:
        model.eval()
    for ax, i in zip(axes.flat, idxs):
        img, tgt = dataset[int(i)]
        pred = predict(img)
        keep = pred["scores"] >= score_thr
        canvas = (img * 255).to(torch.uint8)
        canvas = draw_bounding_boxes(canvas, tgt["boxes"], colors="lime", width=3)
        canvas = draw_bounding_boxes(
            canvas, pred["boxes"][keep].cpu(), colors="red", width=2, font_size=14,
            labels=[f"{classes[l]} {s:.2f}" for l, s in zip(pred["labels"][keep].tolist(), pred["scores"][keep].tolist())],
        )
        ax.imshow(canvas.permute(1, 2, 0))
        ax.set_title(f"test #{i} | GT: {', '.join(classes[l] for l in tgt['labels'].tolist())}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.show()
    return idxs


@torch.no_grad()
def measure_inference(model, dataset, n_warmup=5, predict=None):
    """ms por imagen con batch 1, incluyendo pre y post-procesado del modelo (no la E/S).
    `predict(img)` sustituye a la llamada al modelo (p. ej. TTA: todas las vistas + fusión)."""
    predict = predict or (lambda x: model([x]))
    if model is not None:
        model.eval()
    imgs = [dataset[i][0] for i in range(len(dataset))]
    for img in imgs[:n_warmup]:
        predict(img.to(device))
    times = []
    for img in imgs:
        x = img.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000


# ----------------------------------------------------------------------------
# Fusión de detecciones de varias fuentes (vistas de TTA o modelos de un ensemble)
# ----------------------------------------------------------------------------
def _empty_pred():
    return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}


def nms_fusion(preds_list, iou_thr=0.5, max_dets=100):
    """Junta las detecciones de todas las fuentes y aplica NMS por clase: de cada grupo de cajas
    solapadas se queda la de mayor score, sin modificarla."""
    from torchvision.ops import batched_nms

    boxes = torch.cat([p["boxes"].cpu() for p in preds_list])
    if not len(boxes):
        return _empty_pred()
    scores = torch.cat([p["scores"].cpu() for p in preds_list])
    labels = torch.cat([p["labels"].cpu() for p in preds_list])
    keep = batched_nms(boxes, scores, labels, iou_thr)[:max_dets]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


def wbf(preds_list, iou_thr=0.55, max_dets=100, weights=None):
    """Weighted Boxes Fusion (Solovyev et al., 2021), por clase.

    Las cajas de todas las fuentes se recorren de mayor a menor score y se agrupan: una caja se
    añade al grupo cuya caja fusionada solape con ella con IoU > iou_thr, o abre un grupo nuevo.
    La caja fusionada de cada grupo es la **media de sus cajas ponderada por el score** (a diferencia
    de NMS, que se queda con una sola). Su score es la media de los scores multiplicada por
    min(n_cajas, n_fuentes) / n_fuentes: una detección que solo aparece en algunas fuentes baja.
    `weights` pondera cada fuente (por defecto todas igual).
    """
    from torchvision.ops import box_iou

    n_src = len(preds_list)
    weights = weights or [1.0] * n_src
    boxes = torch.cat([p["boxes"].cpu().float() for p in preds_list])
    if not len(boxes):
        return _empty_pred()
    scores = torch.cat([p["scores"].cpu() * w for p, w in zip(preds_list, weights)])
    labels = torch.cat([p["labels"].cpu() for p in preds_list])

    out_b, out_s, out_l = [], [], []
    for c in labels.unique():
        m = labels == c
        b, s = boxes[m], scores[m]
        clusters, fused = [], []
        for i in s.argsort(descending=True).tolist():
            if fused:
                ious = box_iou(b[i:i + 1], torch.stack(fused))[0]
                j = int(ious.argmax())
                if ious[j] > iou_thr:
                    clusters[j].append(i)
                    idx = torch.tensor(clusters[j])
                    fused[j] = (b[idx] * s[idx, None]).sum(0) / s[idx].sum()
                    continue
            clusters.append([i])
            fused.append(b[i].clone())
        for idx, fb in zip(clusters, fused):
            sc = s[torch.tensor(idx)]
            out_b.append(fb)
            out_s.append(sc.mean() * min(len(idx), n_src) / n_src / (sum(weights) / n_src))
            out_l.append(c)
    scores_t = torch.stack(out_s)
    order = scores_t.argsort(descending=True)[:max_dets]
    return {"boxes": torch.stack(out_b)[order], "scores": scores_t[order], "labels": torch.stack(out_l)[order]}


def predictions_to_mapviz(preds, targets, classes):
    """Listas de predicciones y targets (en el orden del dataset) -> (gts, preds) de mapviz,
    el mismo formato que `collect_predictions`."""
    gts, out = [], []
    for idx, (t, o) in enumerate(zip(targets, preds)):
        img = str(idx)
        for b, l in zip(torch.as_tensor(t["boxes"]).tolist(), t["labels"].tolist()):
            gts.append(dict(img=img, cls=classes[l], box=tuple(b)))
        for b, l, s in zip(o["boxes"].tolist(), o["labels"].tolist(), o["scores"].tolist()):
            out.append(dict(img=img, cls=classes[l], box=tuple(b), score=s))
    return gts, out


# ----------------------------------------------------------------------------
# Análisis de errores (TIDE simplificado de mapviz)
# ----------------------------------------------------------------------------
def import_mapviz():
    import sys

    # mapviz.py está en la raíz del repositorio, compartido por los dos modelos:
    # en local es la carpeta padre; en Kaggle / Colab se descarga al directorio de trabajo
    if Path("../mapviz.py").exists():
        sys.path.insert(0, str(Path("..").resolve()))
    else:
        ensure_file("mapviz.py", repo_path="mapviz.py")
        sys.path.insert(0, ".")
    import mapviz

    return mapviz


@torch.no_grad()
def collect_predictions(model, loader, classes):
    """Cajas reales y predicciones en el formato de mapviz. El loader no debe barajar."""
    model.eval()
    gts, preds, idx = [], [], 0
    for images, targets in loader:
        outputs = model([img.to(device) for img in images])
        for t, o in zip(targets, outputs):
            img = str(idx)
            for b, l in zip(t["boxes"].tolist(), t["labels"].tolist()):
                gts.append(dict(img=img, cls=classes[l], box=tuple(b)))
            for b, l, s in zip(o["boxes"].cpu().tolist(), o["labels"].cpu().tolist(), o["scores"].cpu().tolist()):
                preds.append(dict(img=img, cls=classes[l], box=tuple(b), score=s))
            idx += 1
    return gts, preds


def tide_analysis(split_data, iou_thresholds=(0.5, 0.85), verbose=True):
    """Descomposición TIDE por partición y umbral. `split_data` = {split: (gts, preds)}.
    Devuelve {"<split>@<iou>": resultado de mapviz.error_breakdown}."""
    mv = import_mapviz()
    tide = {f"{name}@{thr}": mv.error_breakdown(gts, preds, iou_thr=thr)
            for name, (gts, preds) in split_data.items() for thr in iou_thresholds}
    if verbose:
        keys = list(mv.ERROR_TYPES) + ["miss"]
        print(f"{'partición@IoU':<14}{'AP base':>9} | " + " | ".join(f"{k:>12}" for k in keys))
        for name, r in tide.items():
            cells = " | ".join(f"{r['counts'][k]:>4d} {r['delta'][k]:>+7.3f}" for k in keys)
            print(f"{name:<14}{r['base']:>9.3f} | {cells}")
    return tide


def tide_to_json(tide):
    return {k: {"base": r["base"], "delta": r["delta"], "counts": r["counts"]} for k, r in tide.items()}


def save_predictions(split_data, out_dir):
    """Guarda gts y predicciones por partición (sirven para TTA, ensembles y la comparativa final)."""
    for name, (gts, preds) in split_data.items():
        _save_json({"gts": gts, "preds": preds}, Path(out_dir) / f"preds_{name}.json")


def load_predictions(name, splits=("valid", "test")):
    """Lee `resultados/<name>/preds_<split>.json` -> {split: (gts, preds)}, o None si no existen."""
    out = {}
    for split in splits:
        try:
            d = _load_json(ensure_file(f"{(RESULTS_ROOT / name).as_posix()}/preds_{split}.json"))
        except FileNotFoundError:
            return None
        to_t = lambda xs: [{**x, "box": tuple(x["box"])} for x in xs]
        out[split] = (to_t(d["gts"]), to_t(d["preds"]))
    return out


# ----------------------------------------------------------------------------
# Figuras del análisis de errores (las mismas que la sección 13 del baseline)
# ----------------------------------------------------------------------------
def show_and_save(plot_fn, out_path, *args, **kwargs):
    """Las funciones de mapviz llaman a plt.show(); se intercepta para poder guardar la figura."""
    original_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        plot_fn(*args, **kwargs)
    finally:
        plt.show = original_show
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_ap_vs_iou(split_data, label, out_dir, references=None):
    """AP en función del umbral de IoU por partición (sección 13.2 del baseline).
    `references` = {etiqueta: split_data de otro experimento}, para superponer sus curvas."""
    mv = import_mapviz()
    for split, (gts, preds) in split_data.items():
        models = {f"{name} ({split})": ref[split][1] for name, ref in (references or {}).items() if ref}
        models[f"{label} ({split})"] = preds
        show_and_save(mv.plot_ap_vs_iou, Path(out_dir) / f"ap_vs_iou_{split}.png", gts, models)


def plot_tide(tide, out_path):
    """Barras de ΔAP por tipo de error, una por partición y umbral (sección 13.3 del baseline)."""
    mv = import_mapviz()
    splits = sorted({k.split("@")[0] for k in tide}, key=["valid", "test"].index)
    thrs = sorted({k.split("@")[1] for k in tide}, key=float)
    fig, axes = plt.subplots(len(splits), len(thrs), figsize=(13, 3.6 * len(splits)), squeeze=False)
    xmax = max(max(r["delta"].values()) for r in tide.values()) * 1.3 + 1e-3
    for i, s in enumerate(splits):
        for j, t in enumerate(thrs):
            r = tide[f"{s}@{t}"]
            mv._bars(axes[i, j], r, xmax=xmax, title=f"{s} | IoU {t} | AP base = {r['base']:.3f}")
            axes[i, j].set_xlabel("ΔAP si se corrige ese tipo de error (nº de errores entre paréntesis)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.show()


def detection_confusion(gts, preds, classes, score_thr=0.5, iou_thr=0.5):
    """Matriz de confusión de detección (sección 13.4 del baseline).

    Cada caja real se empareja con la detección de mayor score (≥ score_thr) que la solape con
    IoU ≥ iou_thr, sea de la clase que sea; si no hay, cuenta como "no detectado". Las detecciones
    sin emparejar cuentan en la fila "fondo". Filas: clase real (+ fondo); columnas: predicha (+ no detectado).
    """
    mv = import_mapviz()
    names = classes[1:]
    n = len(names)
    cm = np.zeros((n + 1, n + 1), dtype=int)
    pos = {c: i for i, c in enumerate(names)}
    for img in sorted({g["img"] for g in gts} | {p["img"] for p in preds}, key=int):
        g_img = [g for g in gts if g["img"] == img]
        p_img = sorted([p for p in preds if p["img"] == img and p["score"] >= score_thr], key=lambda p: -p["score"])
        used = set()
        for g in g_img:
            cand = [(p["score"], k) for k, p in enumerate(p_img) if k not in used and mv.iou(g["box"], p["box"]) >= iou_thr]
            if cand:
                _, k = max(cand)
                used.add(k)
                cm[pos[g["cls"]], pos[p_img[k]["cls"]]] += 1
            else:
                cm[pos[g["cls"]], n] += 1
        for k, p in enumerate(p_img):
            if k not in used:
                cm[n, pos[p["cls"]]] += 1
    return cm


def plot_confusion(split_data, classes, out_path, score_thr=0.5, iou_thr=0.5):
    fig, axes = plt.subplots(1, len(split_data), figsize=(7.5 * len(split_data), 6.5), squeeze=False)
    rows = classes[1:] + ["fondo (FP)"]
    cols = classes[1:] + ["no detectado"]
    cms = {}
    for ax, (name, (gts, preds)) in zip(axes[0], split_data.items()):
        cm = cms[name] = detection_confusion(gts, preds, classes, score_thr, iou_thr)
        ax.imshow(np.log1p(cm), cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    color = ("white" if cm[i, j] > 3 else "black") if i == j else "darkred"
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10, color=color)
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("predicción")
        ax.set_ylabel("clase real")
        ax.set_title(f"{name} (score ≥ {score_thr}, IoU ≥ {iou_thr})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.show()
    return cms


def print_confusion_summary(cms, classes):
    """Resumen en texto de las matrices de `plot_confusion`: aciertos y cada tipo de error."""
    names = classes[1:]
    n = len(names)
    for split, cm in cms.items():
        off = [(names[i], names[j], cm[i, j]) for i in range(n) for j in range(n) if i != j and cm[i, j]]
        fp = [(names[j], cm[n, j]) for j in range(n) if cm[n, j]]
        miss = [(names[i], cm[i, n]) for i in range(n) if cm[i, n]]
        print(f"{split}: {np.trace(cm[:n, :n])} aciertos | {sum(c for *_, c in off)} confusiones entre clases | "
              f"{cm[n, :n].sum()} falsos positivos de fondo | {cm[:n, n].sum()} no detectadas")
        for a, b, c in off:
            print(f"   confusión  {a} -> {b}: {c}")
        for a, c in fp:
            print(f"   fondo detectado como {a}: {c}")
        for a, c in miss:
            print(f"   {a} no detectada: {c}")


def plot_worst_images(gts, preds, tide_result, dataset, out_path, n=6, min_score=0.3):
    """Imágenes con más errores (sección 13.5 del baseline). `tide_result` = tide["test@0.5"]."""
    mv = import_mapviz()
    worst = mv.worst_images(gts, preds, tide_result, n=n)
    images_np = {img: dataset[int(img)][0].permute(1, 2, 0).numpy() for img in worst}
    show_and_save(mv.show_errors, out_path, gts, preds, tide_result, worst,
                  image=images_np, ncols=3, min_score=min_score)
    return worst


def error_analysis_figures(split_data, tide, classes, test_dataset, out_dir, label, references=None):
    """Genera y guarda todas las figuras del análisis de errores del baseline:
    ap_vs_iou_<split>.png, tide.png, matriz_confusion.png y peores_imagenes_test.png."""
    out_dir = Path(out_dir)
    plot_ap_vs_iou(split_data, label, out_dir, references)
    plot_tide(tide, out_dir / "tide.png")
    cms = plot_confusion(split_data, classes, out_dir / "matriz_confusion.png")
    gts, preds = split_data["test"]
    plot_worst_images(gts, preds, tide["test@0.5"], test_dataset, out_dir / "peores_imagenes_test.png")
    return cms


# ----------------------------------------------------------------------------
# Resultados y comparación entre experimentos
# ----------------------------------------------------------------------------
def make_summary(name, description, config, fit_result, res, per_class, times_ms):
    """Mismo esquema que `baseline_results.json`, más el nombre y la descripción del experimento."""
    return {
        "experimento": name,
        "descripcion": description,
        "modelo": "Faster R-CNN v2 (ResNet-50 + FPN), pesos COCO_V1",
        "dataset": DATASET,
        "config": config,
        "mejor_epoca": fit_result["best_epoch"],
        "val_mAP": fit_result["best_map"],
        "test": {
            "mAP@0.5:0.95": res["map"].item(),
            "mAP@0.5": res["map_50"].item(),
            "mAP@0.75": res["map_75"].item(),
            "mAR@100": res["mar_100"].item(),
            "por_clase": per_class,
        },
        "inferencia_ms": float(np.mean(times_ms)),
        "hardware": hardware_name(),
        "tiempo_entrenamiento_min": fit_result["train_time"] / 60,
    }


def save_experiment(out_dir, summary, tide=None):
    out_dir = Path(out_dir)
    _save_json(summary, out_dir / "results.json")
    if tide is not None:
        _save_json(tide_to_json(tide), out_dir / "tide.json")
    print(f"Guardado en {out_dir}/")


def load_experiment(name):
    """Carga results/history/tide de `resultados/<name>/`. Si no están en local (Kaggle, Colab),
    los descarga de `faster_rcnn/resultados/<name>/` del repositorio: los .json sí se suben a git."""
    folder = RESULTS_ROOT / name
    results_file = "baseline_results.json" if name == "baseline_fasterrcnn" else "results.json"
    out = {}
    for key, fname in [("results", results_file), ("history", "history.json"), ("tide", "tide.json")]:
        try:
            out[key] = _load_json(ensure_file(f"{folder.as_posix()}/{fname}"))
        except FileNotFoundError as e:
            print(f"Aviso: {e}")
            out[key] = None
    return out


METRICS = ["mAP@0.5:0.95", "mAP@0.5", "mAP@0.75", "mAR@100"]


def compare_metrics(experiments):
    """Tabla de métricas de test (y val mAP) de varios experimentos; el primero es la referencia.
    `experiments` = {etiqueta: results.json cargado}."""
    names = list(experiments)
    ref = experiments[names[0]]
    w = max(18, *(len(n) for n in names)) + 2
    header = f"{'métrica':<26}" + "".join(f"{n:>{w}}" for n in names)
    print(header)
    print("-" * len(header))

    def row(label, get, fmt="{:.4f}", delta=True):
        vals = [get(e) for e in experiments.values()]
        cells = []
        for k, v in enumerate(vals):
            txt = fmt.format(v)
            if delta and k > 0:
                txt += f" ({v - vals[0]:+.4f})"
            cells.append(f"{txt:>{w}}")
        print(f"{label:<26}" + "".join(cells))

    row("val mAP@0.5:0.95", lambda e: e["val_mAP"])
    for m in METRICS:
        row(f"test {m}", lambda e, m=m: e["test"][m])
    for cls in ref["test"]["por_clase"]:
        row(f"  AP {cls}", lambda e, c=cls: e["test"]["por_clase"][c]["AP"])
    row("inferencia (ms)", lambda e: e["inferencia_ms"], fmt="{:.1f}", delta=False)
    row("entrenamiento (min)", lambda e: e["tiempo_entrenamiento_min"], fmt="{:.1f}", delta=False)
    row("mejor época", lambda e: e["mejor_epoca"], fmt="{:d}", delta=False)


def compare_tide(tides, keys=("valid@0.85", "test@0.85", "test@0.5")):
    """ΔAP por tipo de error en varios experimentos. `tides` = {etiqueta: tide.json cargado}."""
    types = ["loc", "cls", "dup", "bg", "miss"]
    for key in keys:
        print(f"\n=== {key} ===")
        print(f"{'experimento':<26}{'AP base':>9} | " + " | ".join(f"{t:>13}" for t in types))
        for name, tide in tides.items():
            r = tide.get(key) if tide else None
            if r is None:
                continue
            cells = " | ".join(f"{r['counts'][t]:>4d} {r['delta'][t]:>+8.3f}" for t in types)
            print(f"{name:<26}{r['base']:>9.3f} | {cells}")


def plot_history_comparison(histories, out_path=None):
    """Pérdida de validación, mAP de validación y LR por época de varios experimentos."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for name, h in histories.items():
        ep = np.arange(1, len(h["val_map"]) + 1)
        line, = axes[0].plot(ep, h["val_loss"], "-", label=name)
        axes[0].plot(ep, h["train_loss"], ":", c=line.get_color())
        axes[1].plot(ep, h["val_map"], "o-", ms=3, c=line.get_color(), label=name)
        lr = h.get("lr")  # el history del baseline no lo registró (era constante)
        if lr is not None:
            axes[2].plot(ep, lr, "-", c=line.get_color(), label=name)
    axes[0].set_title("Pérdida (continua = validación, punteada = train)")
    axes[1].set_title("mAP@0.5:0.95 en validación")
    axes[2].set_title("Learning rate al inicio de cada época")
    for ax in axes:
        ax.set_xlabel("época")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
    plt.show()
