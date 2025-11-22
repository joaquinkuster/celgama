"""
API Flask para Clasificación de Celulares
==========================================

Este módulo proporciona una API REST para clasificar celulares en gamas
utilizando el modelo K-Means entrenado.

Características principales:
- Soft clustering con cálculo de probabilidades
- Detección de zonas ambiguas entre clusters
- Visualización con PCA
- Análisis de factores determinantes

Endpoints:
- GET /: Página principal
- POST /api/resultado: Clasificación de dispositivo

Autor: Sistema de ML
Fecha: 2025-11-20
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from typing import Dict, List, Tuple, Any, Optional

# ===== CONFIGURACIÓN DE LA APLICACIÓN =====
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clave_secreta_default')

# ===== CONSTANTES =====
# Threshold para determinar si un dispositivo está en zona ambigua
# Si la diferencia de distancias entre el cluster más cercano y el segundo
# más cercano es menor a este porcentaje, se considera ambiguo
AMBIGUITY_THRESHOLD: float = 0.15  # 15%

# Definición de columnas (features) utilizadas por el modelo
COLUMNAS: List[str] = [
    'num_cores',                    # Número de núcleos del procesador
    'processor_speed',              # Velocidad del procesador (GHz)
    'battery_capacity',             # Capacidad de batería (mAh)
    'fast_charging_available',      # Carga rápida disponible (1=Sí, 0=No)
    'ram_capacity',                 # Memoria RAM (GB)
    'internal_memory',              # Almacenamiento interno (GB)
    'screen_size',                  # Tamaño de pantalla (pulgadas)
    'resolution_width',             # Resolución horizontal (píxeles)
    'resolution_height',            # Resolución vertical (píxeles)
    'num_rear_cameras',             # Número de cámaras traseras
    'primary_camera_rear',          # Cámara principal trasera (MP)
    'primary_camera_front',         # Cámara frontal (MP)
    'price'                         # Precio (USD)
]

# ===== CARGA DE MODELOS Y DATOS =====
modelos_cargados: bool = False
scaler = None
modelo_kmeans = None
mapeo_gamas: Dict[int, str] = {}
promedios_clusters: Optional[pd.DataFrame] = None
distribucion_gamas: Dict[str, int] = {}

try:
    # Cargar el escalador (StandardScaler)
    scaler = joblib.load('scaler.pkl')
    
    # Cargar el modelo de clustering K-Means
    modelo_kmeans = joblib.load('modelo_kmeans.pkl')
    
    # Cargar el mapeo de clusters a gamas
    mapeo_gamas = joblib.load('mapeo_gama.pkl')
    
    # Cargar estadísticas de los clusters
    promedios_clusters = pd.read_csv('clusters_promedios.csv')
    
    # Cargar distribución de dispositivos por gama
    distribucion_gamas = joblib.load('distribucion_gamas.pkl')
    
    modelos_cargados = True
    print("✅ Modelos cargados exitosamente")
except Exception as error:
    print(f"❌ Error al cargar modelos: {error}")


# ===== RUTAS =====
@app.route('/')
def index() -> str:
    """
    Renderiza la página principal de la aplicación.
    
    Returns:
        HTML de la página principal
    """
    return render_template('index.html')


@app.route('/api/resultado', methods=['POST'])
def obtener_resultado() -> Tuple[Dict[str, Any], int]:
    """
    Endpoint que recibe las características de un dispositivo y retorna la clasificación.
    
    Proceso:
    1. Valida los datos de entrada
    2. Escala las características
    3. Predice el cluster y calcula probabilidades
    4. Detecta si está en zona ambigua
    5. Calcula factores determinantes
    6. Genera visualización PCA
    
    Returns:
        Tupla con (respuesta JSON, código HTTP)
        
    Respuesta incluye:
        - gama: Gama predicha (Baja, Media, Alta)
        - promedio: Características promedio de la gama
        - factores_clave: Factores que más influyeron
        - dif_relativas: Diferencias por categoría
        - pca_clusters: Coordenadas 2D de centroides
        - pca_usuario: Coordenadas 2D del usuario
        - es_ambiguo: Si está en zona límite entre gamas
        - clusters_cercanos: Clusters dentro del threshold
        - probabilidades: Probabilidad de pertenencia a cada cluster
    """
    # Verificar que los modelos estén cargados
    if not modelos_cargados:
        return jsonify({'error': 'Modelos no disponibles. Ejecuta model.py primero.'}), 500
    
    try:
        # ===== 1. OBTENER Y VALIDAR DATOS =====
        datos_json = request.get_json()
        
        # Validar que todos los campos necesarios estén presentes
        for campo in COLUMNAS:
            if campo not in datos_json or datos_json[campo] == '':
                return jsonify({'error': f'Campo faltante o vacío: {campo}'}), 400
        
        # ===== 2. CONVERTIR DATOS A FORMATO NUMÉRICO =====
        valores_numericos = convertir_datos_a_numericos(datos_json)
        if isinstance(valores_numericos, tuple):  # Error
            return valores_numericos
        
        # Crear DataFrame con los datos del usuario
        datos_usuario = pd.DataFrame([valores_numericos], columns=COLUMNAS)
        
        # ===== 3. PREDECIR CLUSTER Y CALCULAR PROBABILIDADES =====
        datos_escalados = scaler.transform(datos_usuario)
        cluster_predicho = modelo_kmeans.predict(datos_escalados)[0]
        gama_predicha = mapeo_gamas[cluster_predicho]
        
        # ===== 4. SOFT CLUSTERING - CALCULAR DISTANCIAS Y PROBABILIDADES =====
        soft_clustering_info = calcular_soft_clustering(datos_escalados, cluster_predicho)
        
        # ===== 5. OBTENER ESTADÍSTICAS DEL CLUSTER =====
        datos_cluster = promedios_clusters[promedios_clusters['cluster'] == cluster_predicho].iloc[0]
        caracteristicas_promedio = formatear_caracteristicas_promedio(datos_cluster)
        
        # ===== 6. CALCULAR FACTORES CLAVE =====
        factores_determinantes = calcular_factores_clave(
            datos_usuario.iloc[0].to_dict(), 
            datos_cluster
        )
        
        # ===== 7. CALCULAR DIFERENCIAS RELATIVAS =====
        diferencias_relativas = calcular_diferencias_relativas(
            datos_escalados, 
            cluster_predicho
        )
        
        # ===== 8. ANÁLISIS PCA (VISUALIZACIÓN 2D) =====
        pca_info = calcular_pca_visualization(datos_escalados)
        
        # ===== 9. PREPARAR RESPUESTA COMPLETA =====
        respuesta = {
            # Clasificación principal
            'gama': gama_predicha,
            
            # Estadísticas del cluster
            'promedio': caracteristicas_promedio,
            
            # Factores que más influyeron
            'factores_clave': factores_determinantes,
            
            # Diferencias por categoría
            'dif_relativas': diferencias_relativas,
            
            # Datos para visualización PCA
            'pca_clusters': pca_info['clusters_2d'],
            'pca_usuario': pca_info['usuario_2d'],
            'gamas': pca_info['gamas'],
            
            # Información de proximidad (legacy)
            'distancia_minima': soft_clustering_info['distancia_minima'],
            'gama_cercana': soft_clustering_info['gama_cercana'],
            
            # Soft clustering - NUEVO
            'es_ambiguo': soft_clustering_info['es_ambiguo'],
            'clusters_cercanos': soft_clustering_info['clusters_cercanos'],
            'probabilidades': soft_clustering_info['probabilidades'],
            'todas_distancias': soft_clustering_info['todas_distancias'],
            
            # Distribución general
            'distribucion': distribucion_gamas,
            'total_dispositivos': sum(distribucion_gamas.values())
        }
        
        return jsonify(respuesta)
        
    except Exception as error:
        print(f"❌ Error en la predicción: {error}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(error)}), 500


# ===== FUNCIONES AUXILIARES =====

def convertir_datos_a_numericos(datos_json: Dict[str, Any]) -> List[float]:
    """
    Convierte los datos del formulario a valores numéricos.
    
    Args:
        datos_json: Diccionario con los datos del formulario
        
    Returns:
        Lista de valores numéricos en el orden de COLUMNAS
        
    Raises:
        ValueError: Si algún valor no puede ser convertido
    """
    valores_numericos = []
    for columna in COLUMNAS:
        if columna == 'fast_charging_available':
            # Convertir a binario (1 o 0)
            valor = 1.0 if str(datos_json.get(columna, '0')) == '1' else 0.0
        else:
            try:
                valor = float(datos_json[columna])
            except ValueError:
                return jsonify({'error': f'Valor inválido en campo: {columna}'}), 400
        valores_numericos.append(valor)
    
    return valores_numericos


def calcular_soft_clustering(datos_escalados: np.ndarray, cluster_predicho: int) -> Dict[str, Any]:
    """
    Calcula información de soft clustering: distancias a todos los centroides,
    probabilidades de pertenencia, y detecta zonas ambiguas.
    
    Args:
        datos_escalados: Datos del usuario escalados
        cluster_predicho: Cluster predicho por K-Means
        
    Returns:
        Diccionario con información de soft clustering:
        - todas_distancias: Distancias a cada centroide
        - probabilidades: Probabilidad de pertenencia a cada cluster
        - es_ambiguo: Si está en zona ambigua
        - clusters_cercanos: Lista de clusters cercanos
        - distancia_minima: Distancia al cluster más cercano
        - gama_cercana: Gama del cluster más cercano
    """
    # Calcular distancias a todos los centroides
    centroides = modelo_kmeans.cluster_centers_
    distancias = euclidean_distances(datos_escalados, centroides)[0]
    
    # Ordenar distancias para encontrar los más cercanos
    indices_ordenados = np.argsort(distancias)
    distancia_min = distancias[indices_ordenados[0]]
    distancia_segunda = distancias[indices_ordenados[1]]
    
    # Calcular probabilidades usando softmax inverso de distancias
    # Invertir distancias (más cerca = mayor probabilidad)
    distancias_inv = 1.0 / (distancias + 1e-10)  # Evitar división por cero
    probabilidades_raw = distancias_inv / np.sum(distancias_inv)
    
    # Convertir a diccionario con nombres de gamas
    probabilidades = {
        mapeo_gamas[i]: float(prob) 
        for i, prob in enumerate(probabilidades_raw)
    }
    
    # Detectar ambigüedad: si la diferencia relativa entre las dos distancias
    # más cercanas es menor al threshold, es ambiguo
    diferencia_relativa = abs(distancia_min - distancia_segunda) / distancia_min
    es_ambiguo = bool(diferencia_relativa < AMBIGUITY_THRESHOLD)  # Convertir a bool nativo de Python
    
    # Identificar clusters cercanos (dentro del threshold)
    clusters_cercanos = []
    for idx in indices_ordenados:
        diff_rel = abs(distancias[idx] - distancia_min) / distancia_min
        if diff_rel <= AMBIGUITY_THRESHOLD:
            clusters_cercanos.append({
                'cluster_id': int(idx),
                'gama': mapeo_gamas[idx],
                'distancia': float(distancias[idx]),
                'probabilidad': float(probabilidades_raw[idx])
            })
    
    return {
        'todas_distancias': {mapeo_gamas[i]: float(d) for i, d in enumerate(distancias)},
        'probabilidades': probabilidades,
        'es_ambiguo': es_ambiguo,
        'clusters_cercanos': clusters_cercanos,
        'distancia_minima': float(distancia_min),
        'gama_cercana': mapeo_gamas[indices_ordenados[0]]
    }


def formatear_caracteristicas_promedio(datos_cluster: pd.Series) -> Dict[str, Any]:
    """
    Formatea las características promedio del cluster para la respuesta.
    
    Args:
        datos_cluster: Serie con los datos promedio del cluster
        
    Returns:
        Diccionario con características formateadas
    """
    return {
        'Núcleos': int(datos_cluster['num_cores']),
        'Velocidad (GHz)': round(datos_cluster['processor_speed'], 1),
        'Batería (mAh)': int(datos_cluster['battery_capacity']),
        'Carga rápida': 'Sí' if datos_cluster['fast_charging_available'] > 0.5 else 'No',
        'RAM (GB)': int(datos_cluster['ram_capacity']),
        'Almacenamiento (GB)': int(datos_cluster['internal_memory']),
        'Pantalla (pulg)': round(datos_cluster['screen_size'], 1),
        'Ancho de resolución': int(datos_cluster['resolution_width']),
        'Altura de resolución': int(datos_cluster['resolution_height']),
        'Cámaras traseras': int(datos_cluster['num_rear_cameras']),
        'Cámara principal (MP)': int(datos_cluster['primary_camera_rear']),
        'Cámara frontal (MP)': int(datos_cluster['primary_camera_front']),
        'Precio (USD)': f"${int(datos_cluster['price'])}"
    }


def calcular_factores_clave(datos_usuario: Dict[str, float], datos_cluster: pd.Series) -> List[str]:
    """
    Identifica las características con mayor diferencia relativa
    respecto al promedio del cluster.
    
    Args:
        datos_usuario: Diccionario con los datos del usuario
        datos_cluster: Serie con los datos promedio del cluster
        
    Returns:
        Lista con los 5 factores más relevantes ordenados por similitud
    """
    # Mapeo de nombres técnicos a nombres legibles
    mapeo_nombres = {
        'price': 'precio',
        'ram_capacity': 'memoria RAM',
        'primary_camera_rear': 'cámara principal',
        'processor_speed': 'velocidad del procesador',
        'battery_capacity': 'capacidad de batería',
        'internal_memory': 'almacenamiento interno'
    }
    
    diferencias = []
    
    for campo_tecnico, nombre_legible in mapeo_nombres.items():
        valor_usuario = datos_usuario[campo_tecnico]
        valor_promedio = datos_cluster[campo_tecnico]
        
        # Calcular diferencia relativa (evitar división por cero)
        if valor_promedio > 0:
            diferencia_relativa = abs(valor_usuario - valor_promedio) / valor_promedio
            diferencias.append((nombre_legible, diferencia_relativa))
    
    # Ordenar por diferencia (de menor a mayor) y tomar los 5 primeros
    # Los factores con menor diferencia son los más determinantes
    diferencias.sort(key=lambda x: x[1], reverse=False)
    return [factor[0] for factor in diferencias[:5]]


def calcular_diferencias_relativas(datos_usuario_escalados: np.ndarray, cluster: int) -> Dict[str, float]:
    """
    Calcula diferencias relativas agrupadas por categorías técnicas.
    
    Categorías:
    - Procesador: núcleos, velocidad
    - Memoria: RAM, almacenamiento
    - Pantalla: tamaño, resolución
    - Cámara: número de cámaras, megapíxeles
    - Batería: capacidad, carga rápida
    - Precio: costo del dispositivo
    
    Args:
        datos_usuario_escalados: Datos del usuario escalados
        cluster: ID del cluster predicho
        
    Returns:
        Diccionario con diferencias relativas por categoría
    """
    # Definir categorías y sus índices en el array de features
    categorias = {
        'Procesador': [0, 1],          # num_cores, processor_speed
        'Memoria': [4, 5],              # ram_capacity, internal_memory
        'Pantalla': [6, 7, 8],          # screen_size, resolution_width, resolution_height
        'Cámara': [9, 10, 11],          # num_rear_cameras, primary_camera_rear, primary_camera_front
        'Batería': [2, 3],              # battery_capacity, fast_charging_available
        'Precio': [12]                  # price
    }

    # Obtener valores promedio del cluster
    fila_cluster = promedios_clusters[promedios_clusters['cluster'] == cluster].iloc[0]
    valores_cluster = fila_cluster[COLUMNAS].values.reshape(1, -1)

    # Convertir datos escalados de vuelta a escala original
    df_cluster = pd.DataFrame(valores_cluster, columns=COLUMNAS)
    df_usuario = pd.DataFrame(
        scaler.inverse_transform(datos_usuario_escalados), 
        columns=COLUMNAS
    )

    # Calcular diferencias relativas por categoría
    diferencias_por_categoria = {}
    
    for nombre_categoria, indices in categorias.items():
        diferencias_parciales = []
        
        for indice in indices:
            columna = COLUMNAS[indice]
            valor_usuario = df_usuario.iloc[0][columna]
            valor_cluster = df_cluster.iloc[0][columna]
            
            # Calcular diferencia relativa (evitar división por cero)
            if valor_cluster != 0:
                diferencia = (valor_usuario - valor_cluster) / valor_cluster
                diferencias_parciales.append(diferencia)
        
        # Promedio de diferencias en la categoría
        diferencias_por_categoria[nombre_categoria] = float(
            np.mean(diferencias_parciales)
        ) if diferencias_parciales else 0.0

    return diferencias_por_categoria


def calcular_pca_visualization(datos_escalados: np.ndarray) -> Dict[str, Any]:
    """
    Calcula la proyección PCA para visualización 2D.
    
    Args:
        datos_escalados: Datos del usuario escalados
        
    Returns:
        Diccionario con información de PCA:
        - clusters_2d: Coordenadas 2D de los centroides
        - usuario_2d: Coordenadas 2D del usuario
        - gamas: Nombres de las gamas
    """
    # Reducir dimensionalidad para visualización
    pca = PCA(n_components=2)
    centroides_escalados = modelo_kmeans.cluster_centers_
    pca.fit(centroides_escalados)

    # Transformar clusters y usuario al espacio 2D
    clusters_2d = pca.transform(centroides_escalados)
    usuario_2d = pca.transform(datos_escalados)

    return {
        'clusters_2d': clusters_2d.tolist(),
        'usuario_2d': usuario_2d[0].tolist(),
        'gamas': [mapeo_gamas[i] for i in range(len(centroides_escalados))]
    }


# ===== INICIO DEL SERVIDOR =====
if __name__ == '__main__':
    # Crear directorios necesarios
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🚀 SERVIDOR FLASK - CLASIFICADOR DE CELULARES")
    print("=" * 60)
    
    if not modelos_cargados:
        print("⚠️  ADVERTENCIA: Los modelos no están cargados.")
        print("   Ejecuta primero: python model.py")
    else:
        print(f"✅ Modelos cargados correctamente")
        print(f"📊 Distribución de gamas:")
        for gama, cantidad in distribucion_gamas.items():
            print(f"   - {gama}: {cantidad} dispositivos")
        print(f"\n⚙️  Threshold de ambigüedad: {AMBIGUITY_THRESHOLD * 100}%")
    
    print(f"\n🌐 Servidor corriendo en: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    
    # Iniciar servidor en modo desarrollo
    app.run(debug=True, host='127.0.0.1', port=5000)