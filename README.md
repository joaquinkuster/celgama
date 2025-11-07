# 📱 Clasificador de Celulares

Sistema de clasificación inteligente de teléfonos celulares por gama usando Machine Learning (K-Means clustering).

## ✨ Características Principales

- ✅ **Conversión automática de precios**: INR → USD (tasa: 1 USD = 83 INR)
- ✅ **Sin límite de precio**: Acepta cualquier valor desde $50 USD
- ✅ **Análisis de factores determinantes**: Identifica automáticamente los 3 factores más importantes
- ✅ **Gráficos interactivos**: Visualización con matplotlib y análisis PCA real
- ✅ **Dataset real**: ~980 dispositivos sin datos hardcodeados
- ✅ **Navegación con Enter**: Presiona Enter para avanzar/enviar
- ✅ **Barra de progreso mejorada**: El progreso se alinea perfectamente con los círculos de paso
- ✅ **Animación de fondo**: Smartphones flotantes animados
- ✅ **Análisis real**: Todo basado en el dataset procesado

## 🚀 Instalación

### 1. Requisitos previos

```bash
# Python 3.8+
python --version

# Pip actualizado
pip install --upgrade pip
```

### 2. Instalar dependencias

```bash
pip install flask pandas scikit-learn joblib matplotlib seaborn numpy
```

### 3. Estructura de archivos

```
celgama/
├── data/
│   └── celulares.csv          # Dataset con precios en INR
├── templates/
│   └── index.html             # Interfaz web
├── app.py                     # Servidor Flask
├── model.py                   # Entrenamiento del modelo
└── README.md
```

## 📊 Uso

### Paso 1: Entrenar el modelo

```bash
python model.py
```

Este script:
- Carga el dataset de `data/celulares.csv`
- **Convierte precios de INR a USD** (1 USD = 83 INR)
- Limpia datos y elimina valores faltantes
- Entrena modelo K-Means con 3 clusters
- Aplica PCA para visualización
- Genera archivos:
  - `scaler.pkl`
  - `modelo_kmeans.pkl`
  - `mapeo_gama.pkl`
  - `pca.pkl`
  - `distribucion_gamas.pkl`
  - `clusters_promedios.csv`
  - `pca_coords.csv`
  - `dataset_procesado.csv`

**Salida esperada:**
```
==================================================
Entrenando modelo de clasificación de celulares
==================================================

1. Cargando dataset...
   ✓ Dataset cargado: 980 registros

2. Convirtiendo precios de INR a USD (1 USD = 83.0 INR)...
   ✓ Precios convertidos
   - Rango de precios: $12.05 - $301.20 USD

3. Seleccionando características...
4. Verificando valores faltantes...
   ✓ Dataset limpio: 980 registros

...

✅ Modelo entrenado y guardado exitosamente
```

### Paso 2: Iniciar servidor

```bash
python app.py
```

**Salida esperada:**
```
==================================================
🚀 Iniciando servidor Flask...
==================================================

📊 Dataset: 980 dispositivos
🌐 URL: http://localhost:5000
==================================================
```

### Paso 3: Usar la aplicación

1. Abre tu navegador en `http://localhost:5000`
2. Haz clic en "Comenzar"
3. Completa los 6 pasos del formulario
4. **Tip**: Presiona **Enter** para avanzar entre pasos o enviar
5. Visualiza el resultado con gráficos interactivos

## 🎯 Factores Determinantes

El sistema identifica automáticamente los 3 factores más importantes que determinan la gama comparando tu dispositivo con el promedio del cluster:

**Factores analizados:**
- Precio (USD)
- Memoria RAM (GB)
- Cámara principal (MP)
- Velocidad del procesador (GHz)
- Capacidad de batería (mAh)
- Almacenamiento interno (GB)
- Núcleos del procesador
- Tamaño de pantalla (pulgadas)

**Algoritmo:**
1. Calcula la desviación relativa de cada característica respecto al promedio del cluster
2. Ordena por desviación descendente
3. Retorna los top 3 factores con mayor diferencia

## 📈 Gráficos Generados

### Gráfico 1: Comparación con Promedio de Gama
Compara 6 categorías principales:
- Procesador
- Memoria
- Pantalla
- Cámara
- Batería
- Precio

### Gráfico 2: Distribución de Dispositivos (PCA)
Muestra los **980 dispositivos reales** del dataset en un scatter plot 2D usando PCA:
- Cada punto = un dispositivo del dataset
- Colores por gama (Baja/Media/Alta)
- Tu dispositivo marcado con estrella roja
- Cantidades reales por gama

## 🎨 Mejoras Visuales

### Barra de Progreso
- ✅ Progreso alineado perfectamente con círculos de paso
- ✅ Animación suave entre pasos
- ✅ Indicadores de completado

### Fondo Animado
- 6 iconos de smartphones flotantes
- Animación infinita con movimiento vertical y rotación
- Opacidad reducida (10%) para no distraer

### Navegación
- **Enter**: Avanzar al siguiente paso
- **Enter** en último paso: Ver resultado
- **Enter** en pantalla de resultados: Realizar otra evaluación
- Botones tradicionales también disponibles

## 🔧 Configuración Avanzada

### Cambiar tasa de conversión INR→USD

Edita `model.py` línea 24:
```python
INR_TO_USD = 83.0  # Cambiar según tasa actual
```

### Ajustar número de clusters

Edita `model.py` línea 56:
```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# Cambiar n_clusters a 4 o 5 para más gamas
```

### Personalizar colores de gama

Edita `app.py` función `generar_grafico_personalizado`:
```python
colores_gama = {
    'Gama Baja': '#ff6b6b',   # Rojo
    'Gama Media': '#ffd93d',  # Amarillo
    'Gama Alta': '#6bcf7f'    # Verde
}
```

## 📝 Campos del Formulario

### Paso 1: Procesador
- Núcleos: 1-16
- Velocidad: 1.0-4.0 GHz

### Paso 2: Batería
- Capacidad: 1000-10000 mAh
- Carga rápida: Sí/No

### Paso 3: Memoria
- RAM: 1-16 GB
- Almacenamiento: 8-1024 GB

### Paso 4: Pantalla
- Tamaño: 4.0-7.5 pulgadas
- Resolución: 720x1280 hasta 3840x2160 px

### Paso 5: Cámara
- Cámaras traseras: 1-5
- Cámara principal: 8-200 MP
- Cámara frontal: 5-50 MP

### Paso 6: Precio
- Precio: desde $50 USD (sin límite superior)

## 🐛 Solución de Problemas

### Error: "No such file or directory: 'data/celulares.csv'"
**Solución**: Asegúrate de que el archivo CSV esté en la carpeta `data/`

### Error: "No such file or directory: 'scaler.pkl'"
**Solución**: Ejecuta primero `python model.py` para generar los modelos

### Los gráficos no se muestran
**Solución**: Verifica que matplotlib esté instalado con backend Agg:
```bash
pip install matplotlib --upgrade
```

### Error al enviar formulario
**Solución**: 
1. Verifica que todos los campos estén llenos
2. Revisa la consola del navegador (F12) para ver errores
3. Verifica que Flask esté corriendo en el puerto 5000

## 📊 Estadísticas del Dataset

Después de ejecutar `model.py`, verás estadísticas como:

```
Total de dispositivos: 980

Distribución por gama:
  - Gama Alta: 327 dispositivos (33.4%)
  - Gama Baja: 326 dispositivos (33.3%)
  - Gama Media: 327 dispositivos (33.3%)

Estadísticas por gama:

Gama Baja:
  - Procesador: 4 núcleos @ 1.8 GHz
  - RAM: 3 GB
  - Almacenamiento: 32 GB
  - Batería: 3500 mAh
  - Cámara principal: 13 MP
  - Precio promedio: $100 USD

... (más estadísticas)
```

## 🚀 Características Técnicas

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn (K-Means, PCA, StandardScaler)
- **Visualización**: matplotlib, seaborn
- **Frontend**: HTML5, CSS3, JavaScript (Anime.js)
- **Base de datos**: ~980 dispositivos reales
- **Conversión de moneda**: INR → USD automática

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo licencia MIT.

## 👨‍💻 Autor

Desarrollado para clasificación inteligente de dispositivos móviles.

---

**¿Preguntas?** Revisa el código o abre un issue en el repositorio.