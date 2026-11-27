"""common: código compartido por los cuadernos de Faster R-CNN v2 sobre pills-sxdht.

El cuaderno del baseline (`baseline-faster-r-cnn-v2.ipynb`) define todo el pipeline en línea.
Aquí está el mismo código, sin cambios de comportamiento, para que cada cuaderno de mejora
solo contenga lo que cambia respecto al baseline. Así todos los experimentos usan exactamente
la misma limpieza de clases, los mismos DataLoaders y la misma evaluación.

Contenido:
    - Datos: carga de Hugging Face, limpieza de clases, `PillsDataset`, DataLoaders.
    - Modelo: `build_model` (Faster R-CNN v2 COCO_V1 con cabeza nueva).
    - Entrenamiento: `train_one_epoch`, `validation_loss`, `evaluate`, `fit`, schedulers.
    - Evaluación en test: `test_report`, `plot_pr_curves`, `measure_inference`.
    - Análisis de errores: `collect_predictions`, `tide_analysis` (usa `mapviz.py`).
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
    res = evaluate(model, loader, extended=True)
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
def measure_inference(model, dataset, n_warmup=5):
    """ms por imagen con batch 1, incluyendo pre y post-procesado del modelo (no la E/S)."""
    model.eval()
    imgs = [dataset[i][0] for i in range(len(dataset))]
    for img in imgs[:n_warmup]:
        model([img.to(device)])
    times = []
    for img in imgs:
        x = img.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model([x])
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000


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
