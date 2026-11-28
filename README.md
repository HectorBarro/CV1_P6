# Practica deteccion de objetos - pills-sxdht

Dataset `Francesco/pills-sxdht` (Hugging Face) con 7 clases: `Cipro 500`, `Ibuphil Cold 400-60`, `Xyzall 5mg`, `blue`, `pink`, `red` y `white`.
Quitamos `pills` (no tiene cajas) e `Ibuphil 600 mg` (solo aparece en la imagen 12 de train y la 61 de test, que se eliminan).

## YOLOv10-S (carpeta `yolov10/`)

| fichero | que hace | donde se ejecuto |
|---|---|---|
| `yolov10/baseline_yolov10.ipynb` | fine-tuning de `yolov10s.pt` con todo por defecto, evaluacion en test (ultralytics y `mapviz`) y analisis TIDE | Colab, GPU T4 (las celdas de `mapviz` y TIDE, en local) |
| `yolov10/mejoras_yolov10.ipynb` | experimentos de mejora (E1-E5), TTA en todos los modelos, modelo final, test, TIDE antes y despues y conclusiones | Kaggle, GPU T4 (la celda de errores de clase, en local con los mismos pesos) |
| `mapviz.py` (en la raiz) | funciones del profesor (P1) para calcular el mAP y el TIDE, compartido con Faster R-CNN | |

### Como ejecutarlo

1. **Baseline**: abrir `yolov10/baseline_yolov10.ipynb` en Colab con GPU T4, subir `mapviz.py` y ejecutar todo (unos 15 min).
   El dataset se descarga solo y se pasa a formato YOLO. El modelo queda en `runs/detect/yolov10s_baseline/weights/best.pt`.
2. **Mejoras**: en Kaggle con GPU T4, subir como dataset `mapviz.py` y el `best.pt` del baseline, y ejecutar todo `yolov10/mejoras_yolov10.ipynb` (unas 2 h, entrena E1, E2, E3 y E5). La penultima celda comprime `runs/` en `runs_mejoras.zip`, que se descarga desde la pestaña Output.

Semilla `seed=0` en todo. En GPU puede haber pequeñas diferencias entre ejecuciones.
**Todas las comparaciones (entre experimentos y con Faster R-CNN) se hacen con `mapviz.py`**, con score minimo 0.05 (el valor por defecto de Faster R-CNN en torchvision). El mAP de ultralytics se calcula de otra forma y cambia entre versiones, asi que no se mezcla con el de `mapviz`.

### Resultados

Ultralytics guarda todo en `runs/`. De ahi copiamos solo lo necesario a `yolov10/resultados/` (no esta en git):
- `baseline/` y `e1_epochs200/` (modelo final): `best.pt`, `results.png` (curvas de entrenamiento) y `results.csv`
- `e2_box10_dfl2/`, `e3_imgsz960/`, `e5_todo/`: curvas de entrenamiento
- `baseline_test/`: curva PR y matriz de confusion del baseline en test (ultralytics)

mAP50-95 con `mapviz`:

| modelo | val | test |
|---|---|---|
| baseline | 0.905 | 0.906 |
| final (E1, 200 epocas, + TTA con volteo) | 0.946 | 0.902 |

En val el final gana, pero en test empata con el baseline. El resto de experimentos y la explicacion estan en las conclusiones de `yolov10/mejoras_yolov10.ipynb`.
