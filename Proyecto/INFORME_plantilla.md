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

- **Dataset:** PlantVillage (`tensorflow_datasets`), subconjunto de **hojas de tomate**.
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

Promedio de los `[N_SPLITS]` folds. La desviación estándar refleja la estabilidad de cada
configuración frente al azar del reparto e inicialización.

| Configuración | Accuracy | Precision | Recall | F1 (macro) | Tiempo (s) | Params entrenables |
|---|---|---|---|---|---|---|
| A — Congelado | `[... ± ...]` | `[... ± ...]` | `[... ± ...]` | `[... ± ...]` | `[... ± ...]` | `[...]` |
| B — Fine-tuning | `[... ± ...]` | `[... ± ...]` | `[... ± ...]` | `[... ± ...]` | `[... ± ...]` | `[...]` |

### 4.2 Detalle por fold

| Fold | F1 A | F1 B | Tiempo A (s) | Tiempo B (s) |
|---|---|---|---|---|
| 1 | `[...]` | `[...]` | `[...]` | `[...]` |
| 2 | `[...]` | `[...]` | `[...]` | `[...]` |
| `[...]` | `[...]` | `[...]` | `[...]` | `[...]` |

### 4.3 Desempeño vs. costo
_(Insertar figura: barras de F1 (macro) y tiempo con barras de error media ± std.)_
**Figura 1.** F1 vs. tiempo de entrenamiento (media ± std entre folds). `[descripción breve]`

### 4.4 Curvas de aprendizaje (fold representativo)
_(Insertar figura: Loss/Accuracy por época de Config A y Config B del último fold.)_
**Figura 2.** Curvas de aprendizaje de un fold. `[descripción breve]`

### 4.5 Matrices de confusión (agregadas out-of-fold)
_(Insertar figuras de las matrices de confusión de ambas configuraciones, construidas con las
predicciones out-of-fold de todos los folds.)_
**Figura 3.** Matrices de confusión agregadas. `[descripción breve]`

### 4.6 Ejemplos bien y mal clasificados
_(Insertar mosaico de imágenes correctas e incorrectas del fold representativo.)_
**Figura 4.** Ejemplos cualitativos. `[descripción breve]`

---

## 5. Análisis e interpretación

- **Aprovechamiento del Transfer Learning:** `[¿la validación subió rápido y se estabilizó alta?]`
- **Sobreajuste:** `[¿hubo brecha train–val? ¿en qué fase? ¿magnitud?]`
- **Comparación A vs. B:** la Configuración B `[mejoró / no mejoró]` el F1 en `[Δ...]` puntos. Esto
  se explica porque `[al descongelar, el modelo adaptó características específicas de hojas... / la
  línea base ya capturaba bien las características y el ajuste fino aportó poco]`.
- **Estabilidad y significancia (validación cruzada):** la desviación estándar del F1 fue
  `[±... (A)]` y `[±... (B)]`. Los intervalos (media ± std) de A y B `[se solapan / no se solapan]`,
  por lo que la diferencia observada `[no es concluyente / es consistente entre folds]`. La
  variación entre folds fue `[baja / alta]`, lo que indica que el resultado `[es estable / depende
  bastante del reparto de datos]`.
- **Errores frecuentes (matriz de confusión):** las clases más confundidas fueron
  `[clase X ↔ clase Y]`, probablemente por `[similitud visual de síntomas]`.

---

## 6. Recomendaciones (juicio de ingeniería basado en evidencia)

- **Configuración recomendada:** `[A / B]`, porque `[justificación con números]`.
- **Relación desempeño/costo:** la mejora de `[ΔF1]` frente a un costo de `[Δtiempo]` es
  `[razonable / no justificada]` en el escenario de `[...]`.
- **Condiciones de uso adecuadas:** `[...]`.
- **Limitaciones:** `[dataset de laboratorio con fondo uniforme; posible caída de desempeño en
  fotos reales de campo; clases con pocas muestras; etc.]`.
- **Mejoras futuras:** `[EfficientNet/ResNet; más data augmentation; descongelar más capas;
  recolectar imágenes de campo; aumentar N_SPLITS o repetir con varias semillas; balanceo de
  clases]`.

---

## 7. Conclusiones

- `[Respuesta directa a la pregunta experimental: ¿se confirmó H1?]`
- `[Hallazgo principal sobre desempeño vs. costo.]`
- `[Aprendizaje clave sobre transfer learning vs. fine-tuning.]`

---

## 8. Referencias

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted
   Residuals and Linear Bottlenecks.* CVPR.
2. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health
   (PlantVillage).* arXiv:1511.08060.
3. Deng, J., et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR.
4. Abadi, M., et al. (2016). *TensorFlow: A System for Large-Scale Machine Learning.*
5. `[Notas del curso Sistemas Inteligentes II — Transfer Learning, Prof. J. A. Jaramillo Garzón.]`
