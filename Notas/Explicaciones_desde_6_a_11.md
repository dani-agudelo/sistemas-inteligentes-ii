# Explicaciones (1 - Introducción a 11 - Regresión)

Formato usado en cada notebook:
- **El "Por qué" conceptual** (intuición y geometría/analogía)
- **Explicación paso a paso** (bloques del código/idea)
- **El "Secreto para el parcial/examen"** (resumen corto técnico)

---

## Notas/1 - Introducción al aprendizaje de máquina.ipynb

### El "Por qué" conceptual
El aprendizaje supervisado parte de una idea muy humana: aprendes de ejemplos. La máquina “ve” un conjunto de entrenamiento, encuentra patrones y luego intenta generalizar (extrapolar) a casos nuevos.

En este notebook te enseñan un caso base: un **clasificador discriminante lineal**. Piensa en una recta (en 2D) o un hiperplano (en más dimensiones) que separa clases. El vector `w` es “la brújula” que apunta **perpendicularmente** a la frontera, y la predicción sale del signo de `w^T x`.

Para construir esa frontera usan **mínimos cuadrados**: elige los pesos para que el error entre predicción y etiqueta sea lo más pequeño posible.

### Explicación paso a paso
1. **Entender qué está aprendiendo el modelo**
   - Cada muestra es una fila de la matriz `X` y cada característica es una columna.
   - La frontera se define con una función discriminante lineal:
     - Si `w^T x = 0`, estás “en” la frontera.
     - Si `w^T x > 0` vs `w^T x < 0`, caes en una u otra clase.

2. **Elegir etiquetas tipo “+1 / -1”**
   - El notebook usa una convención típica para trabajar con `sign` y mínimos cuadrados, por ejemplo reemplazar `0` por `-1`.

3. **Incluir sesgo (bias)**
   - En vez de manejarlo aparte, “lo metes” agregando una columna de unos:
     - `Xm = np.hstack((X, np.ones((N, 1))))`

4. **Minimizar el error por mínimos cuadrados**
   - Se define el error:
     - `ε = (1/2) * Σ (w^T x_i - y_i)^2`
   - La solución cerrada (con bias incluido) queda del tipo:
     - `w = (X^T X)^-1 X^T y`
   - En el notebook aparece como:
     - `XTX = np.transpose(Xm).dot(Xm)`
     - `XTXi = np.linalg.inv(XTX)`
     - `wm = ((XTXi).dot(np.transpose(Xm))).dot(y)`

5. **Predecir con `sign`**
   - Para una muestra `test`, la predicción es el signo:
     - `prediccion = np.sign(wm.dot(np.transpose(test)))`

6. **Equivalente práctico con `sklearn`**
   - Usan `RidgeClassifier(alpha=0)` para obtener la solución tipo mínimos cuadrados de forma directa:
     - `clf = linear_model.RidgeClassifier(alpha=0)`
     - `clf.fit(X[0:100,:2], y[0:100])`
     - `clf.predict(test)`

### El "Secreto para el parcial/examen"
- Frontera lineal: **`w^T x = 0`** y la predicción sale del **`sign(w^T x)`**.
- Mínimos cuadrados: **`ε = (1/2) Σ (w^T x_i - y_i)^2`**.
- Solución cerrada (con bias): **`w = (X^T X)^-1 X^T y`**.
- `RidgeClassifier(alpha=0)` es el atajo “de sklearn” para ese caso (sin regularización).

---

## Notas/2 - Clasificadores_generativos y validación.ipynb

### El "Por qué" conceptual
Aquí el foco es aprender por **similitud** y “mirar al vecindario”.

En vez de construir una frontera con una fórmula fija (como un discriminante lineal), un clasificador tipo **K-NN** (vecinos más cercanos) dice: *“a este punto nuevo lo clasifico según cómo son los puntos más parecidos que ya conozco”*.

Para saber qué es “parecido”, necesitas una **métrica de distancia** (el notebook usa distancia Euclídea). Y luego, para decidir la clase, haces **votación** entre los `k` vecinos más cercanos.

Pero ojo: el modelo solo sirve si generaliza. Por eso el notebook dedica bastante a cómo evaluar (hold-out, bootstrapping y cross-validation con *folds*), y luego usar métricas desde la **matriz de confusión**.

### Explicación paso a paso
1. **Similitud = distancia Euclídea**
   - Usan distancia inducida por norma 2:
     - `distancia = lambda x, y: np.sqrt(sum((x-y)**2))`

2. **Regla del Nearest Neighbor (y luego K-NN)**
   - Calculas distancias desde el punto de prueba hacia todos los puntos del entrenamiento.
   - Ordenas por menor distancia:
     - `orden = D.argsort()`
   - Tomas los `k` primeros vecinos:
     - `V_k = X[orden[0:k]]`
   - Votación (clase más frecuente):
     - `votos = y[orden[0:k]]`
     - `statistics.mode(votos)`

3. **K-NN con `sklearn`**
   - Implementación directa:
     - `clf = KNeighborsClassifier(n_neighbors=7)`
     - `clf.fit(X, y)`
     - `clf.predict(test)`

4. **Validación: medir generalización**
   - **Hold-out**: separas datos en entrenamiento vs prueba (por ejemplo 80/20 o 70/30).
   - **Bootstrapping**: repites muchos hold-out (remuestreo con reemplazo) y miras media y desviación del rendimiento.
   - **Cross-validation**:
     - divides en `n` *folds*,
     - entrenas `n` veces dejando cada *fold* distinto como prueba.

5. **Matriz de confusión y métricas**
   - Con `confusion_matrix` obtienes `TP, FP, TN, FN`:
     - `cm = confusion_matrix(ytest, predicciones)`
   - Métricas:
     - Sensibilidad / Recall: `TP / (TP + FN)`
     - Especificidad: `TN / (TN + FP)`

### El "Secreto para el parcial/examen"
- K-NN = *“memoria” + distancia + votación*.
- Tu métrica clave aquí es Euclídea (y `k` controla cuánta información local usas).
- No basta con accuracy: con `confusion_matrix` puedes responder Recall (sensibilidad) y Especificidad.
- Las validaciones (hold-out, bootstrapping, cross-validation con *folds*) reducen el sesgo de una sola partición.

---

## Notas/3 - Inferencia bayesiana.ipynb

### El "Por qué" conceptual
La inferencia bayesiana te da una forma ordenada de responder: *“¿qué es lo más probable ahora que ya vi los datos?”*.

La idea central es **Bayes**:
- **prior**: lo que creías antes de ver los datos.
- **likelihood / verosimilitud**: qué tan bien explica los datos un modelo/hipótesis.
- **evidence**: normaliza para que el resultado sea una probabilidad.
- **posterior**: lo que crees después de ver los datos.

Además, el notebook contrasta **discriminativos vs generativos**:
- discriminativo: modela directamente la frontera de decisión,
- generativo: modela la distribución de los datos para cada clase y deja que la frontera “aparezca” desde esas probabilidades.

Luego conectan esto con estimación de parámetros:
- con prior uniforme obtienes **MLE** (máxima verosimilitud), que termina siendo mínimos cuadrados,
- con un prior Gaussiano obtienes **MAP**, que lleva a **Ridge regression** (regularización).

### Explicación paso a paso
1. **Usar el teorema de Bayes**
   - El notebook escribe la relación:
     - `posterior = (verosimilitud * prior) / evidencia`

2. **Clasificación generativa**
   - construyes un modelo para los datos de cada clase (por ejemplo con una distribución Gaussiana),
   - eliges la clase que hace al posterior más grande.

3. **Likelihood con Gaussiana (caso continuo)**
   - Cuando los datos son continuos, se usa una densidad (comúnmente Gaussiana) como verosimilitud.

4. **MLE: prior uniforme**
   - Si el prior es uniforme, maximizar el posterior equivale a maximizar la verosimilitud.
   - Para hacerlo numéricamente mejor usan **log-verosimilitud**.
   - Derivar respecto a `w`, igualar a cero y despejar:
     - el resultado lleva a **regresión lineal por mínimos cuadrados**.

5. **MAP: prior Gaussiano y regularización**
   - Asumen un prior:
     - `p(w) ~ N(0, Σ_p)`
   - Eso produce una posterior sobre `w` cuya solución se escribe como:
     - `w_bar = (1/σ^2) * ( (X X^T)/σ^2 + Σ_p^{-1} )^{-1} X y`
   - Caso especial: si `Σ_p = k I`, aparece **Ridge regression**.

6. **Naive Bayes Gaussiano como ejemplo práctico**
   - Librería:
     - `from sklearn.naive_bayes import GaussianNB`
     - `clf = GaussianNB()`
     - `clf.fit(X, y)`
     - `clf.predict(test)`
   - Y, a nivel conceptual, estiman medias y covarianzas con:
     - `np.mean(...)` y `np.cov(...)`.

### El "Secreto para el parcial/examen"
- Bayes siempre: **posterior = likelihood * prior / evidence**.
- Generativo vs discriminativo: generativo modela `p(x|clase)` y de ahí sale la frontera.
- MLE (prior uniforme) => coincide con mínimos cuadrados.
- MAP (prior Gaussiano) => **Ridge** (regularización por prior sobre pesos).

---

## Notas/4 - Perceptrón-SVM.ipynb

### El "Por qué" conceptual
Perceptrón y SVM son **clasificadores lineales**: buscan un hiperplano que separe clases.

La diferencia está en “qué significa mejor”:
- El **Perceptrón** intenta corregirse cuando se equivoca (actualiza en base a muestras mal clasificadas).
- El **SVM** quiere algo más: además de separar, busca el **mayor margen**, es decir, el hiperplano que queda lo más lejos posible de las muestras “críticas”.

En SVM esas muestras críticas son los **support vectors**, y en la formulación dual aparecen como términos con `lambda_i` distinto de cero.

### Explicación paso a paso
1. **Función discriminante lineal**
   - El notebook usa la forma:
     - `f(x) = w^T x`
   - `w` es el vector normal a la frontera y `x` recorre las muestras.
   - Se incluye sesgo típicamente agregando una columna de unos a `X` (creando `Xm`).

2. **Criterio de error del Perceptrón**
   - Definen conjuntos de muestras mal clasificadas usando que el producto:
     - `w^T x_i y_i` debe ser positivo si va bien.
   - El error se basa en las que hacen que ese producto sea negativo:
     - `I = { i | w^T x_i * y_i < 0 }`

3. **Actualización (gradiente descendente estocástico)**
   - Toma una actualización del tipo:
     - `wnuevo = w + alpha * derivada`
   - Donde la derivada usa la suma de “empujes” de las muestras que fallaron:
     - `derivada = Σ Xm[errores[i]] * y[errores[i]]`

4. **Perceptrón con `sklearn` (comparación)**
   - Usan:
     - `clf = Perceptron(tol=0, eta0=0.01)`
     - `clf.fit(...)`
     - `clf.predict(...)`

5. **SVM: maximizar margen como optimización cuadrática**
   - El notebook plantea el objetivo con variables de Lagrange `lambda_i`.
   - Aparece la función (en forma Lagrangiana):
     - `L = (1/2) w^T w - Σ λ_i [ y_i (w^T x_i - 1) ]`
   - Se pasa al problema dual (con restricciones):
     - `Σ y_i λ_i = 0`
     - `λ_i >= 0`

6. **Resolver el dual (programación cuadrática)**
   - Arman la QP con matrices/vectores `P` y `q`.
   - Ejecutan con `qpsolvers.solve_qp`:
     - `lambdas = qpsolvers.solve_qp(P=P, q=q, lb=np.zeros(len(Xm)), solver=solver_use)`
   - Reconstruyen el vector óptimo:
     - `wsvm = Σ Xm[i] * y[i] * lambdas[i]`

### El "Secreto para el parcial/examen"
- Perceptrón: aprende con actualizaciones cuando `w^T x_i` no respeta la etiqueta `y_i` (mala clasificación).
- SVM: el “valor” está en el margen; en dual aparecen `lambda_i`.
- Las `lambda_i` grandes/no-cero corresponden a **support vectors** (los que sostienen la frontera).
- Si te preguntan fórmulas: Perceptrón usa criterio por signo; SVM usa dual con restricciones `Σ y_i λ_i = 0` y `λ_i >= 0`.

---

## Notas/5 - SVM-Kernel.ipynb

### El "Por qué" conceptual
Cuando una frontera lineal no alcanza, el truco es cambiar el punto de vista.

En vez de intentar separar con una recta “en el espacio original”, puedes:
1) mapear los datos a un espacio de más dimensión con un **mapeo** `φ(x)`,
2) ahí buscar una frontera lineal,
3) pero evitando calcular explícitamente `φ(x)` usando el **kernel trick**.

El notebook explica el kernel así: es una función `k(a,b)` que calcula el producto punto en el espacio mapeado, sin que tú tengas que construir ese espacio a mano.

Luego muestran dos kernels típicos:
- kernel **cuadrático** (mapeo con monomios),
- kernel **RBF Gaussiano**, que mide similitud según distancia.

### Explicación paso a paso
1. **Mapeo explícito cuadrático (ejemplo)**
   - Para puntos `x = [x1, x2]`, usan:
     - `φ(x) = [x1^2, x2^2, sqrt(2) x1 x2]`
   - En código aparece como `phiX` con `x^2` y el término cruzado.

2. **Agregar bias para el discriminante**
   - Con:
     - `Xm = np.hstack((phiX, np.ones((len(phiX), 1))))`

3. **Kernel trick (evitar `φ` explícito)**
   - El notebook muestra cómo el producto en el espacio mapeado se puede expresar como:
     - una función del producto punto original,
   - y así defines un **kernel** que reemplaza el cálculo del producto en el espacio nuevo.

4. **Kernel cuadrático (polinomial)**
   - La idea es que al mapear con monomios, el clasificador termina siendo equivalente a una SVM con un kernel polinomial (en el caso de ese mapeo).

5. **Kernel Gaussiano / RBF**
   - Definición:
     - `k(a,b) = exp( -||a-b||^2 / (2 σ^2) )`
   - Implementación típica del notebook:
     - `sigma = 0.1`
     - calcular distancias `D_all`
     - `K = np.exp(-(D_all**2) / (2 * sigma**2))`
   - Para una nueva muestra se calcula `k(x,z)` con expresiones del tipo:
     - `kxz = np.exp(-(D**2) / (2 * sigma**2)).ravel()`

### El "Secreto para el parcial/examen"
- Kernel trick: reemplaza `φ(a)^T φ(b)` por `k(a,b)` sin construir `φ`.
- Kernel = “qué tan distinto” se considera parecido (y por eso define la forma de la frontera).
- RBF: controlas la suavidad con `σ` (o equivalente `gamma`); valores distintos cambian cuán flexible es la frontera.

---

## Notas/6 - ANN.ipynb

### El "Por qué" conceptual

Una neurona puede verse como: *una regla* que combina entradas con pesos y luego aplica una **activación**. El problema es que una sola neurona (sin no linealidad fuerte) tiende a producir decisiones “demasiado rectas”.

La gracia de una **Red Neuronal Artificial (ANN)** es que, al apilar neuronas (capas) con activaciones no lineales (por ejemplo `tanh`, sigmoide, ReLU), la red aprende **una frontera de decisión o función objetivo lo bastante flexible** como para ajustarse a patrones no lineales.

### Explicación paso a paso

1. **De lineal a no lineal**
   - Un perceptrón simple usa una función tipo:
     - combinación lineal (pesos + sesgo)
     - después una activación (no lineal)
   - El notebook remarca que puedes reemplazar el `sign` por una activación **derivable**, porque si quieres aprender con gradiente necesitas derivadas.

2. **Activaciones típicas**
   - `tanh(x)`: útil porque es suave, satura y es derivable.
   - `sigmoid(x)` / sigmoidal: también derivable, pero puede saturar más.
   - Idea clave: la activación mete no linealidad para que el modelo no sea “solo una recta”.

3. **Backpropagation (gradiente descendente)**
   - Objetivo: minimizar un error (pérdida) entre predicción y etiqueta.
   - Backprop hace esto en dos movimientos:
     - **Forward**: calcula salidas de capas.
     - **Backward**: calcula cómo cambia el error si mueves cada peso (regla de la cadena).

4. **Ejemplo de entrenamiento manual (MLP)**
   - El cuaderno muestra una MLP con dos matrices de pesos (por ejemplo `W1` y `W2`).
   - Bloques del entrenamiento (conceptualmente):
     - Inicializar pesos aleatoriamente.
     - Forward:
       - `d = tanh(X @ W1)` (salida oculta)
       - `s = tanh(d @ W2)` (salida final)
     - Error y gradientes:
       - derivar respecto de cada conjunto de pesos
       - actualizar pesos con `W := W - alpha * gradW`
     - Repetir hasta que el error mejore lo suficiente.

5. **Visualización de la frontera de decisión**
   - Se evalúa la red sobre una malla de puntos en el plano y se colorea qué clase predice cada punto.
   - Esto te permite “ver” que con más capas la red logra fronteras curvas y separaciones que una recta no haría.

6. **Perceptrón multicapa con sklearn (comparación)**
   - El notebook también usa `MLPClassifier` para que veas el mismo concepto usando una librería:
     - eliges arquitectura con `hidden_layer_sizes=...`
     - eliges activación con `activation='tanh'` (u otra)
     - entrenas con `fit`
     - y predices con `predict` para la malla.

### El "Secreto para el parcial/examen"

- Una ANN = composición de funciones no lineales: capas + activaciones derivables.
- **Aprendizaje**: backpropagation calcula el gradiente del error y actualiza pesos con gradiente descendente.
- En clasificación, una ANN termina produciendo probabilidades o una salida a interpretar; lo importante es que la no linealidad permite fronteras complejas.
- Lo que te preguntan casi siempre: *por qué es necesario que la activación sea derivable* (porque entrenas con gradiente).

---

## Notas/8-Decision_Trees.ipynb

### El "Por qué" conceptual

Un árbol de decisión es como hacer preguntas en cascada.

- Cada nodo pregunta por un atributo.
- Cada rama lleva a una región del espacio de datos.
- Al final, en una hoja, ya “resuelves” con una clase (clasificación) o con un valor (regresión).

La diferencia con modelos tipo regresión/ANN:
- aquí no estás ajustando directamente una función continua,
- estás **particionando el espacio** (y eso funciona muy bien con variables categóricas o reglas interpretables).

### Explicación paso a paso

1. **ID3: entropía e información**
   - Se usa **entropía de Shannon** para medir impureza/motivo de incertidumbre.
   - Si en un nodo solo hay una clase, la entropía es 0.
   - Si hay mezcla de clases, la entropía aumenta.

   - Luego aparece la **ganancia de información (Information Gain)**:
     - IG = entropía antes de dividir - entropía ponderada después de dividir.

   - Algoritmo ID3 (resumen):
     1. calcular entropía del conjunto actual
     2. para cada atributo, calcular la ganancia de información
     3. escoger el atributo con mayor ganancia
     4. repetir recursivamente en los subconjuntos

2. **Caso manual: Lenses (gafas)**
   - El notebook aplica ID3 a un dataset con atributos categóricos.
   - El resultado es un árbol que termina con reglas del tipo:
     - “si el paciente tiene X y Y entonces necesita lentes duros/blandos/no necesita”.

3. **CART**
   - CART divide en **binario** en cada nodo: elige `(x_j, t)` y crea dos ramas (<= t y > t).
   - El criterio de impureza depende del tipo:
     - clasificación: **Gini**
     - regresión: **varianza residual**
   - Luego hace poda por complejidad (para no sobreajustar).

4. **Control de complejidad en CART**
   En `scikit-learn`, el notebook resalta que se controla con:
   - `max_depth`: limita la profundidad
   - `min_samples_split`: no dividir si hay muy pocos datos
   - `ccp_alpha`: poda por complejidad del costo (penaliza árboles grandes)

5. **Random Forest**
   - Random Forest = muchos CART entrenados con:
     - bootstrap de muestras (bagging)
     - muestreo aleatorio de variables (en cada split)
   - Predicción:
     - clasificación: voto mayoritario
     - regresión: promedio
   - Beneficio clave: reduce la varianza del modelo respecto a un solo árbol.

6. **Titanic con sklearn**
   - El notebook usa Titanic (y normalmente `OneHotEncoder` + `ColumnTransformer` para variables categóricas).
   - Entrena:
     - un árbol (CART)
     - un Random Forest
   - Compara rendimiento y muestra interpretabilidad vía `feature importance` u otros indicadores.

### El "Secreto para el parcial/examen"

- **ID3**: usa entropía y ganancia de información para decidir el atributo.
- **CART**: divide en binario; para clasificación usa **Gini**, y se poda con `ccp_alpha`.
- **Random Forest**: ensamble de CART con bagging y aleatoriedad de features; mejora generalización.
- Parámetros típicos a recordar: `max_depth`, `min_samples_split`, `ccp_alpha`.

---

## Notas/10-Clustering.ipynb

Este notebook recorre algoritmos de clustering (aprendizaje no supervisado) en un hilo común: descubrir estructura en datos sin etiquetas.

### El "Por qué" conceptual

Clustering agrupa muestras por **similitud**:
- sin etiquetas,
- sin “verdad” externa,
- el agrupamiento depende del algoritmo y de cómo mides similitud/vecindad.

El resultado: “islas naturales” en el espacio de características.

### Explicación paso a paso

1. **Clustering jerárquico (dendrograma)**
   - Construye un árbol fusionando clusters sucesivamente.
   - `linkage` calcula cómo se fusionan (definido por un criterio: single/complete/average).
   - Para obtener `k` clusters, se corta el dendrograma con `fcluster(..., criterion='distance')`.

2. **J4 (validación interna)**
   - El notebook calcula J4 como razón tipo Fisher:
     - dispersión entre clusters (SB)
     - sobre dispersión dentro de clusters (SW)
   - Idea: buen clustering = clusters separados (SB grande) y compactos (SW pequeño).

3. **K-Means (centroides)**
   - Requiere fijar `k`.
   - Itera:
     1. asignar puntos al centroide más cercano
     2. actualizar centroides con la media de su cluster
     3. repetir hasta converger.

4. **Clustering espectral (grafo + Laplaciano)**
   - Construye un grafo:
     - nodos = puntos
     - pesos = similitud (por ejemplo con un kernel gaussiano)
   - Calcula el Laplaciano y sus autovectores:
     - usa los autovectores asociados a los autovalores más pequeños
   - Luego aplica K-Means en el espacio embebido.
   - Esto permite separar clusters no convexos (por ejemplo, `make_moons`).

5. **DBSCAN (densidad)**
   - No usa `k`.
   - Parámetros:
     - `eps`: radio de vecindad
     - `min_samples`: umbral de densidad para ser “core”.
   - Clasifica puntos:
     - core
     - border
     - noise (etiqueta `-1`)

### El "Secreto para el parcial/examen"

- Jerárquico: corta dendrograma para elegir número de clusters.
- K-Means: necesita `k` y asume clusters “bien comportados” (convexos/esféricos); minimiza la dispersión intra-cluster por iteración.
- Espectral: Laplaciano + autovectores + K-Means en embedding; logra no convexidad.
- DBSCAN: densidad con `eps` y `min_samples`, detecta ruido con `-1` y no necesita `k`.

Comparación rápida:
- Si sabes `k` y forma es convexa: K-Means.
- Si forma es rara/no convexa: espectral o DBSCAN.
- Si hay ruido/outliers: DBSCAN.

---

## Notas/11-Regresión.ipynb

### El "Por qué" conceptual

Regresión predice una cantidad continua:
- clasificación: etiquetas discretas
- regresión: valores reales (precio, temperatura, etc.)

Aquí el notebook usa un dataset “seno + ruido”, lo que fuerza a modelos no lineales.

### Explicación paso a paso

1. **Marco general**
   - Busca una función `f(x; θ)` para minimizar error esperado.
   - En práctica minimiza error empírico (pérdida MSE):
     - `L(y, ŷ) = (y - ŷ)^2`

2. **Regresión lineal**
   - Baseline:
     - aprende una recta
   - Buena cuando la relación es aproximadamente lineal.
   - Con seno + ruido, subajusta (no puede modelar curvatura completa).

3. **Regresión K-NN**
   - No aprende parámetros explícitos.
   - Para un punto nuevo `x`:
     - encuentra los `k` vecinos más cercanos
     - predice como promedio (o promedio ponderado).
   - Captura no linealidad local, pero no extrapola y depende fuertemente de `k`.

4. **ε-SVR (Support Vector Regression)**
   - Busca una función que caiga dentro de un “tubo” de tolerancia ±ε alrededor de los datos.
   - Errores dentro del tubo no se penalizan; fuera sí.
   - Usa kernels (por ejemplo RBF) para trabajar en espacios de mayor dimensión.
   - Parámetros clave:
     - `epsilon` (tubo)
     - `C` (penaliza violaciones)
     - el kernel (p.ej. RBF con `gamma` en implementaciones típicas)

5. **Regresión con ANN**
   - En regresión la **salida debe ser lineal** (identidad) para producir valores reales.
   - Las capas internas sí usan no linealidades (por ejemplo ReLU).
   - Entrena típicamente con MSE.

### El "Secreto para el parcial/examen"

- MSE como función de pérdida estándar en regresión.
- Regresión lineal: recta, solución cerrada y limitada si la relación no es lineal.
- K-NN regresión: promedia vecinos; depende de `k` y no extrapola.
- ε-SVR: tubo ±ε, penaliza fuera del tubo con `C`, usa kernels (RBF) para no linealidad.
- ANN regresión: activación lineal en salida + no linealidad interna (ReLU/tanh) y MSE.

