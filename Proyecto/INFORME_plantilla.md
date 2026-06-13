# Aprendizaje por Transferencia para la Clasificación de Enfermedades en Hojas de Tomate
### Proyecto Final — Sistemas Inteligentes II · Universidad de Caldas
**Profesor:** Jorge Alberto Jaramillo Garzón
**Integrantes:** _(nombres — máximo 4)_
**Fecha:** _(...)_

> Plantilla lista para rellenar. Reemplaza todo lo que está entre `[...]` con los resultados de tu
> ejecución del notebook `proyecto_final_vision.ipynb`. Cada sección corresponde a un criterio de
> evaluación del enunciado.

---

## 1. Introducción y pregunta experimental

La detección temprana de enfermedades en cultivos es clave para la seguridad alimentaria. El
diagnóstico manual es lento y requiere experticia, por lo que la clasificación automática de
imágenes de hojas mediante visión por computador es una alternativa de alto valor práctico. Sin
embargo, entrenar redes convolucionales desde cero exige grandes volúmenes de datos y cómputo; el
**aprendizaje por transferencia** permite reutilizar modelos preentrenados (en ImageNet) y
adaptarlos a esta tarea con recursos modestos.

**Pregunta experimental.**
> ¿El descongelamiento parcial (fine-tuning) de las últimas capas de MobileNetV2 mejora el
> F1-score frente a usar el modelo únicamente como extractor de características (capas congeladas),
> y cuál es el costo computacional adicional de esa mejora?

**Hipótesis (H1).** El fine-tuning parcial obtendrá un F1-score macro superior al del extractor de
características congelado, a costa de mayor tiempo de entrenamiento y mayor riesgo de sobreajuste.

---

## 2. Descripción del dataset y particiones

- **Dataset:** PlantVillage (carpeta local `Plant_leaf_diseases_dataset_without_augmentation`), subconjunto de **hojas de tomate**.
- **Número de clases:** `[NUM_CLASSES]` (tomate sano + enfermedades: `[listar clases]`).
- **Total de imágenes utilizadas:** `[...]` (submuestreo de `[MAX_PER_CLASS]` por clase).
- **Tamaño de entrada:** `160 × 160 × 3`.
- **Esquema de validación — validación cruzada estratificada (`StratifiedKFold`):**
  en lugar de un único reparto fijo, el experimento se repite en `[N_SPLITS]` folds. En cada fold,
  el fold reservado actúa como **conjunto de prueba** y el resto se usa para entrenar; de ese resto
  se aparta un **15 %** como validación interna para *EarlyStopping*. Las métricas finales se
  reportan como **media ± desviación estándar** entre folds.

| Esquema | Folds (`N_SPLITS`) | Prueba por fold | Validación interna | Estratificada |
|---|---|---|---|---|
| K-Fold CV | `[N_SPLITS]` | ~`[100/N_SPLITS] %` | 15 % del entrenamiento | Sí |

- **Motivación:** un único split fijo puede dar una diferencia A vs. B que dependa del azar del
  reparto. La validación cruzada reduce ese sesgo y permite cuantificar la **estabilidad** de cada
  configuración mediante la desviación estándar.
- **Preprocesamiento:** redimensionado a 160×160 y normalización con `preprocess_input` de
  MobileNetV2 (rango [-1, 1]).
- **Aumento de datos:** `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)` (solo en
  entrenamiento).

---

## 3. Metodología

- **Modelo base:** MobileNetV2 preentrenada en ImageNet (`include_top=False`).
- **Clasificador (head):** `GlobalAveragePooling2D → Dropout(0.2) → Dense(softmax)`.
- **Función de pérdida:** entropía cruzada categórica dispersa.
- **Optimizador:** Adam.
- **Hiperparámetros principales:**

| Hiperparámetro | Valor |
|---|---|
| Batch size | 32 |
| Learning rate (head) | 1e-3 |
| Learning rate (fine-tuning) | 1e-5 |
| Épocas (head) | `[EPOCHS_HEAD]` |
| Épocas (fine-tuning) | `[EPOCHS_FT]` |
| Capa de descongelamiento (`fine_tune_at`) | `[120]` |
| Semilla aleatoria | 42 |
| N.º de folds (`N_SPLITS`) | `[N_SPLITS]` |

- **Hardware:** `[CPU/GPU detectada]`.
- **Reproducibilidad:** semilla fija en `random`, `numpy` y `tensorflow`; los folds de la
  validación cruzada se generan con `StratifiedKFold(shuffle=True, random_state=42)`.
- **Evaluación:** en cada fold se entrenan **ambas** configuraciones sobre las mismas particiones
  (comparación justa) y se promedian las métricas de los `[N_SPLITS]` folds.

### 3.1 Configuraciones comparadas

| | Config A — Feature Extractor | Config B — Fine-Tuning parcial |
|---|---|---|
| Backbone | Congelado | Descongelado desde la capa `[120]` |
| Capas entrenadas | Solo el head | Head + bloque superior del backbone |
| Estrategia | 1 fase | 2 fases (calentamiento + fine-tuning) |
| LR | 1e-3 | 1e-3 → 1e-5 |

La Configuración A actúa como **línea base de referencia** para juzgar si el fine-tuning produce
una mejora real.

---

## 4. Resultados

### 4.1 Métricas agregadas (validación cruzada, media ± desviación estándar)

Promedio de los 5 folds (validación cruzada estratificada). Hardware: CPU, Python 3.11.9,
TensorFlow 2.21.0.

| Configuración | Accuracy | Precision | Recall | F1 (macro) | Tiempo (s) | Params entrenables |
|---|---|---|---|---|---|---|
| A — Congelado | 0.802 ± 0.021 | 0.823 ± 0.020 | 0.802 ± 0.021 | **0.799 ± 0.022** | 306.0 ± 161.8 | 12.810 |
| B — Fine-tuning | 0.799 ± 0.016 | 0.812 ± 0.017 | 0.799 ± 0.016 | 0.796 ± 0.017 | 412.6 ± 135.2 | 1.637.706 |

### 4.2 Detalle por fold

| Fold | F1 A | F1 B | Tiempo A (s) | Tiempo B (s) |
|---|---|---|---|---|
| 1 | 0.830 | 0.808 | 627.8 | 681.5 |
| 2 | 0.820 | 0.820 | 256.7 | 332.0 |
| 3 | 0.779 | 0.776 | 208.5 | 354.4 |
| 4 | 0.784 | 0.782 | 220.2 | 367.2 |
| 5 | 0.779 | 0.792 | 216.7 | 327.8 |

### 4.3 Desempeño vs. costo
_(Insertar figura del notebook: barras de F1 y tiempo con error media ± std.)_
**Figura 1.** Config A alcanza F1 ligeramente superior con ~35% menos tiempo de entrenamiento.

### 4.4 Curvas de aprendizaje (fold representativo)
_(Insertar figura del último fold.)_
**Figura 2.** Validación sube rápido en fase inicial (transfer learning efectivo); en Config B la
brecha train–val se amplía levemente tras el descongelamiento en la época 12.

### 4.5 Matrices de confusión (agregadas out-of-fold)
_(Insertar matrices OOF del notebook.)_
**Figura 3.** Confusiones frecuentes entre enfermedades con síntomas visuales similares (blight,
spot, virus).

### 4.6 Ejemplos bien y mal clasificados
_(Insertar mosaicos del fold 5, Config B.)_
**Figura 4.** Errores asociados a imágenes con síntomas tempranos o iluminación ambigua.

---

## 5. Análisis e interpretación

- **Aprovechamiento del Transfer Learning:** Sí. F1 macro ~0.80 entrenando solo 12.810 parámetros
  del head indica que los filtros ImageNet de MobileNetV2 son discriminativos para hojas de tomate.
- **Sobreajuste:** Config A muestra brecha train–val moderada y EarlyStopping efectivo. Config B
  presenta mayor brecha tras descongelar capas (época 12), sin ganancia en test.
- **Comparación A vs. B:** Config B **no mejoró** el F1; la diferencia media es −0.003 a favor de A.
  El backbone congelado ya captura características suficientes; el fine-tuning adapta train/val sin
  generalizar mejor.
- **Estabilidad (CV):** F1 std = ±0.022 (A) y ±0.017 (B). Los intervalos se solapan; la diferencia
  **no es concluyente**. Folds 1–2 (F1 ~0.82) son más fáciles que 3–5 (~0.78).
- **Errores frecuentes:** Early blight ↔ Septoria leaf spot; Late blight ↔ Target Spot;
  Leaf Mold ↔ Spider mites; mosaic virus ↔ Yellow Leaf Curl Virus.

---

## 6. Recomendaciones (juicio de ingeniería basado en evidencia)

- **Configuración recomendada:** **A (congelado)**, por F1 igual o superior (0.799 vs 0.796),
  ~35% menos tiempo (~306 s vs ~413 s por fold) y ~128× menos parámetros entrenables.
- **Relación desempeño/costo:** ΔF1 = −0.003 y Δt ≈ +35% → el fine-tuning **no está justificado**
  para despliegue con recursos limitados.
- **Condiciones de uso:** Prototipos y clasificación en condiciones controladas (fondo uniforme).
- **Limitaciones:** Dataset de laboratorio, submuestreo 300/clase, CPU, una semilla.
- **Mejoras futuras:** EfficientNet/ResNet, dataset completo, imágenes de campo, multi-seed.

---

## 7. Conclusiones

- **H1 no se confirma:** el fine-tuning parcial no superó al extractor congelado en F1 macro.
- **Hallazgo principal:** MobileNetV2 como feature extractor es la opción más eficiente en este
  escenario (mejor relación desempeño/costo).
- **Aprendizaje clave:** Más parámetros entrenables no implica mejor generalización; el transfer
  learning con backbone congelado puede ser suficiente cuando el dominio es visualmente cercano a
  ImageNet y el dataset es limitado.

---

## 8. Referencias

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted
   Residuals and Linear Bottlenecks.* CVPR.
2. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health
   (PlantVillage).* arXiv:1511.08060.
3. Deng, J., et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR.
4. Abadi, M., et al. (2016). *TensorFlow: A System for Large-Scale Machine Learning.*
5. `[Notas del curso Sistemas Inteligentes II — Transfer Learning, Prof. J. A. Jaramillo Garzón.]`
