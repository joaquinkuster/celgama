from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

# ===== CONFIGURACIÓN DE LA APLICACIÓN =====
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clave_secreta_default')

# ===== DEFINICIÓN DE COLUMNAS (FEATURES) =====
# Estas son todas las características que el modelo utiliza para clasificar
COLUMNAS = [
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
modelos_cargados = False
try:
    # Cargar el escalador (StandardScaler)
    scaler = joblib.load('scaler.pkl')
    
    # Cargar el modelo de clustering K-Means
    modelo_kmeans = joblib.load('modelo_kmeans.pkl')
    
    # Cargar el mapeo de clusters a gamas (ej: {0: 'Gama Baja', 1: 'Gama Media', 2: 'Gama Alta'})
    mapeo_gamas = joblib.load('mapeo_gama.pkl')
    
    # Cargar estadísticas de los clusters
    promedios_clusters = pd.read_csv('clusters_promedios.csv')
    
    # Cargar distribución de dispositivos por gama
    distribucion_gamas = joblib.load('distribucion_gamas.pkl')
    
    modelos_cargados = True
    print("✅ Modelos cargados exitosamente")
except Exception as error:
    print(f"❌ Error al cargar modelos: {error}")

# ===== RUTA PRINCIPAL =====
@app.route('/')
def index():
    """Renderiza la página principal de la aplicación"""
    return render_template('index.html')

# ===== API DE CLASIFICACIÓN =====
@app.route('/api/resultado', methods=['POST'])
def obtener_resultado():
    """
    Endpoint que recibe las características de un dispositivo y retorna:
    - La gama predicha (Baja, Media, Alta)
    - Características promedio de esa gama
    - Factores clave que determinaron la clasificación
    - Datos para visualización (gráficos)
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

        # Crear DataFrame con los datos del usuario
        datos_usuario = pd.DataFrame([valores_numericos], columns=COLUMNAS)
        
        # ===== 3. PREDECIR CLUSTER Y GAMA =====
        # Escalar los datos (normalización)
        datos_escalados = scaler.transform(datos_usuario)
        
        # Predecir el cluster
        cluster_predicho = modelo_kmeans.predict(datos_escalados)[0]
        
        # Obtener la gama correspondiente al cluster
        gama_predicha = mapeo_gamas[cluster_predicho]
        
        # ===== 4. OBTENER ESTADÍSTICAS DEL CLUSTER =====
        datos_cluster = promedios_clusters[promedios_clusters['cluster'] == cluster_predicho].iloc[0]
        
        # Crear diccionario con características promedio de la gama
        caracteristicas_promedio = {
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
        
        # ===== 5. CALCULAR FACTORES CLAVE =====
        factores_determinantes = calcular_factores_clave(
            datos_usuario.iloc[0].to_dict(), 
            datos_cluster
        )
        
        # ===== 6. CALCULAR DIFERENCIAS RELATIVAS =====
        diferencias_relativas = calcular_diferencias_relativas(
            datos_escalados, 
            cluster_predicho
        )
        
        # ===== 7. ANÁLISIS PCA (VISUALIZACIÓN 2D) =====
        # Reducir dimensionalidad para visualización
        pca = PCA(n_components=2)
        centroides_escalados = modelo_kmeans.cluster_centers_
        pca.fit(centroides_escalados)

        # Transformar clusters y usuario al espacio 2D
        clusters_2d = pca.transform(centroides_escalados)
        usuario_2d = pca.transform(datos_escalados)

        # Calcular distancias del usuario a cada cluster
        distancias = euclidean_distances(usuario_2d, clusters_2d)[0]
        cluster_mas_cercano = int(np.argmin(distancias))
        distancia_minima = round(float(distancias[cluster_mas_cercano]), 3)
        gama_mas_cercana = mapeo_gamas[cluster_mas_cercano]
        
        # ===== 8. PREPARAR RESPUESTA COMPLETA =====
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
            'pca_clusters': clusters_2d.tolist(),
            'pca_usuario': usuario_2d[0].tolist(),
            'gamas': [mapeo_gamas[i] for i in range(len(centroides_escalados))],
            
            # Información de proximidad
            'distancia_minima': distancia_minima,
            'gama_cercana': gama_mas_cercana,
            
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


# ===== FUNCIÓN: IDENTIFICAR FACTORES CLAVE =====
def calcular_factores_clave(datos_usuario, datos_cluster):
    """
    Identifica las características con mayor diferencia relativa
    respecto al promedio del cluster.
    
    Retorna los 5 factores más relevantes ordenados por importancia.
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
    diferencias.sort(key=lambda x: x[1], reverse=False)
    return [factor[0] for factor in diferencias[:5]]


# ===== FUNCIÓN: CALCULAR DIFERENCIAS POR CATEGORÍA =====
def calcular_diferencias_relativas(datos_usuario_escalados, cluster):
    """
    Calcula diferencias relativas agrupadas por categorías técnicas.
    
    Categorías:
    - Procesador: núcleos, velocidad
    - Memoria: RAM, almacenamiento
    - Pantalla: tamaño, resolución
    - Cámara: número de cámaras, megapíxeles
    - Batería: capacidad, carga rápida
    - Precio: costo del dispositivo
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


# ===== INICIO DEL SERVIDOR =====
if __name__ == '__main__':
    # Crear directorios necesarios
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 SERVIDOR FLASK - CLASIFICADOR DE CELULARES")
    print("="*60)
    
    if not modelos_cargados:
        print("⚠️  ADVERTENCIA: Los modelos no están cargados.")
        print("   Ejecuta primero: python model.py")
    else:
        print(f"✅ Modelos cargados correctamente")
        print(f"📊 Distribución de gamas:")
        for gama, cantidad in distribucion_gamas.items():
            print(f"   - {gama}: {cantidad} dispositivos")
    
    print(f"\n🌐 Servidor corriendo en: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Iniciar servidor en modo desarrollo
    app.run(debug=True, host='127.0.0.1', port=5000)