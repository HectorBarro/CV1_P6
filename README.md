# Practica deteccion de objetos - pills-sxdht

Dataset `Francesco/pills-sxdht` de Hugging Face, con 7 clases: `Cipro 500`, `Ibuphil Cold 400-60`, `Xyzall 5mg`, `blue`, `pink`, `red` y `white`.
Quitamos `pills` (no tiene cajas) e `Ibuphil 600 mg` (solo sale en la imagen 12 de train y la 61 de test, que quitamos enteras).

`mapviz.py` (el del profesor) esta en la raiz y lo usamos los dos detectores para calcular el mAP y el TIDE.

## YOLOv10-S (carpeta `yolov10/`)

| notebook | que hace | donde |
|---|---|---|
| `baseline_yolov10.ipynb` | fine-tuning de `yolov10s.pt` con todo por defecto, evaluacion en test y TIDE | Colab, GPU T4 |
| `mejoras_yolov10.ipynb` | mejoras E1-E5, TTA, modelo final, test, TIDE antes y despues y conclusiones | Kaggle, GPU T4 |

### Como ejecutarlo

1. **Baseline**: abrir `baseline_yolov10.ipynb` en Colab con GPU T4, subir `mapviz.py` y ejecutar todo (unos 15 min). El dataset se descarga solo y se pasa a formato YOLO. El modelo queda en `runs/detect/yolov10s_baseline/weights/best.pt`.
2. **Mejoras**: en Kaggle con GPU T4, subir como dataset `mapviz.py` y el `best.pt` del baseline y ejecutar todo `mejoras_yolov10.ipynb` (unas 2 h). Al final se crea `runs_mejoras.zip` con los entrenamientos, que se descarga desde Output.

Usamos `seed=0`. En GPU puede salir algo distinto entre ejecuciones.
Para comparar (entre experimentos y con Faster R-CNN) usamos siempre el mAP de `mapviz.py` con score minimo 0.05, no el de ultralytics.

### Resultados

Lo que guarda ultralytics en `runs/` lo copiamos a `yolov10/resultados/` (no se sube a git):
- `baseline/` y `e1_epochs200/`: `best.pt` y curvas de entrenamiento (`results.png`, `results.csv`)
- `e2_box10_dfl2/`, `e3_imgsz960/`, `e5_todo/`: curvas de entrenamiento
- `baseline_test/`: curva PR y matriz de confusion del baseline en test

mAP50-95 con `mapviz`:

| modelo | val | test |
|---|---|---|
| baseline | 0.905 | 0.906 |
| final (E1 + TTA) | 0.946 | 0.902 |

En val el final gana, pero en test se queda igual que el baseline. La explicacion esta en las conclusiones de `mejoras_yolov10.ipynb`.
