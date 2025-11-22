# Métricas de Evaluación de Clustering

Este documento explica las métricas utilizadas para evaluar la calidad del modelo de clustering K-Means.

## Métricas Implementadas

### 1. Silhouette Score (Coeficiente de Silueta)

**¿Qué mide?**
Mide qué tan similar es un objeto a su propio cluster (cohesión) comparado con otros clusters (separación).

**Rango de valores:**
- **-1 a +1**
- Valores cercanos a **+1**: El objeto está bien emparejado con su propio cluster y mal emparejado con clusters vecinos
- Valores cercanos a **0**: El objeto está en el límite entre dos clusters
- Valores cercanos a **-1**: El objeto probablemente fue asignado al cluster incorrecto

**Interpretación:**
- **> 0.7**: Estructura fuerte
- **0.5 - 0.7**: Estructura razonable
- **0.3 - 0.5**: Estructura débil pero aceptable
- **< 0.3**: No hay estructura sustancial

**Fórmula:**
Para cada punto i:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Donde:
- `a(i)` = distancia promedio entre i y todos los demás puntos en el mismo cluster
- `b(i)` = distancia promedio más pequeña entre i y puntos en cualquier otro cluster

---

### 2. Davies-Bouldin Index

**¿Qué mide?**
Mide la relación promedio entre la dispersión dentro de los clusters y la separación entre clusters.

**Rango de valores:**
- **0 a ∞**
- Valores más **bajos** son mejores
- **0** indica clustering perfecto (clusters completamente separados)

**Interpretación:**
- **< 1.0**: Excelente separación entre clusters
- **1.0 - 2.0**: Buena separación
- **> 2.0**: Clusters pueden estar superpuestos, considerar revisar

**Fórmula:**
```
DB = (1/k) * Σ max(R_ij)
```
Donde:
- `k` = número de clusters
- `R_ij` = (S_i + S_j) / d_ij
- `S_i` = dispersión promedio del cluster i
- `d_ij` = distancia entre centroides de clusters i y j

---

### 3. Calinski-Harabasz Score (Variance Ratio Criterion)

**¿Qué mide?**
Ratio entre la dispersión entre clusters y la dispersión dentro de clusters.

**Rango de valores:**
- **0 a ∞**
- Valores más **altos** son mejores
- Indica clusters más densos y mejor separados

**Interpretación:**
- **> 300**: Muy bueno
- **100 - 300**: Aceptable
- **< 100**: Débil, considerar revisar número de clusters

**Fórmula:**
```
CH = (SSB / SSW) * ((n - k) / (k - 1))
```
Donde:
- `SSB` = suma de cuadrados entre clusters (Between-cluster dispersion)
- `SSW` = suma de cuadrados dentro de clusters (Within-cluster dispersion)
- `n` = número total de puntos
- `k` = número de clusters

---

### 4. Inertia (Inercia)

**¿Qué mide?**
Suma de las distancias cuadradas de cada punto a su centroide más cercano.

**Rango de valores:**
- **0 a ∞**
- Valores más **bajos** son mejores
- El valor depende de la escala de los datos

**Interpretación:**
- No hay umbrales absolutos (depende del dataset)
- Útil para comparar diferentes configuraciones del mismo dataset
- Se usa en el método del codo (elbow method) para determinar k óptimo

**Fórmula:**
```
Inertia = Σ min(||x_i - μ_j||²)
```
Donde:
- `x_i` = punto de datos i
- `μ_j` = centroide del cluster j
- La suma es sobre todos los puntos

---

## Cómo Interpretar los Resultados Conjuntamente

### Clustering de Alta Calidad
- Silhouette Score: > 0.5
- Davies-Bouldin Index: < 1.0
- Calinski-Harabasz Score: > 300

### Clustering Aceptable
- Silhouette Score: 0.3 - 0.5
- Davies-Bouldin Index: 1.0 - 2.0
- Calinski-Harabasz Score: 100 - 300

### Clustering Problemático (requiere revisión)
- Silhouette Score: < 0.3
- Davies-Bouldin Index: > 2.0
- Calinski-Harabasz Score: < 100

## Acciones Recomendadas según Métricas

### Si las métricas son malas:

1. **Ajustar el número de clusters (k)**
   - Probar con k=2, k=4, k=5
   - Usar método del codo con Inertia
   - Usar Silhouette Score para diferentes valores de k

2. **Revisar las features**
   - Eliminar features irrelevantes o redundantes
   - Aplicar PCA para reducción de dimensionalidad
   - Normalizar/escalar de manera diferente

3. **Verificar balance de datos**
   - Si hay desbalance severo (>30%), considerar:
     - SMOTE para sobremuestreo
     - Undersampling de la clase mayoritaria
     - Ajustar pesos de las muestras

4. **Probar otros algoritmos**
   - DBSCAN (para clusters de forma irregular)
   - Hierarchical Clustering
   - Gaussian Mixture Models

## Ejemplo de Salida

```json
{
  "metricas_evaluacion": {
    "silhouette_score": 0.4523,
    "davies_bouldin_index": 1.2341,
    "calinski_harabasz_score": 245.67
  },
  "interpretaciones": {
    "silhouette": "Aceptable",
    "davies_bouldin": "Bueno",
    "calinski": "Aceptable"
  },
  "balance": {
    "es_balanceado": true,
    "desbalance": 0.12
  }
}
```

## Referencias

- Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"
- Davies, D. L.; Bouldin, D. W. (1979). "A Cluster Separation Measure"
- Caliński, T.; Harabasz, J. (1974). "A dendrite method for cluster analysis"

---

**Nota**: Estas métricas están implementadas en `model.py` y se guardan automáticamente en `metricas_clustering.json` después del entrenamiento.
