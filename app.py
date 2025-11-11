from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clave_secreta_default')

# Columnas del modelo
COLUMNAS = [
    'num_cores', 'processor_speed', 'battery_capacity',
    'fast_charging_available', 'ram_capacity', 'internal_memory',
    'screen_size', 'resolution_width', 'resolution_height',
    'num_rear_cameras', 'primary_camera_rear', 'primary_camera_front', 'price'
]

# Cargar modelos
modelos_ok = False
try:
    scaler = joblib.load('scaler.pkl')
    modelo = joblib.load('modelo_kmeans.pkl')
    mapeo = joblib.load('mapeo_gama.pkl')
    promedios = pd.read_csv('clusters_promedios.csv')
    distribucion = joblib.load('distribucion_gamas.pkl')
    modelos_ok = True
    print("✅ Modelos cargados")
except Exception as e:
    print(f"❌ Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/resultado', methods=['POST'])
def resultado_api():
    if not modelos_ok:
        return jsonify({'error': 'Modelos no cargados'}), 500
    
    try:
        data = request.get_json()
        
        # Validar campos
        for field in COLUMNAS:
            if field not in data or data[field] == '':
                return jsonify({'error': f'Campo faltante: {field}'}), 400
        
        # Convertir a valores numéricos
        valores = []
        for col in COLUMNAS:
            if col == 'fast_charging_available':
                valor = 1.0 if str(data.get(col, '0')) == '1' else 0.0
            else:
                try:
                    valor = float(data[col])
                except ValueError:
                    return jsonify({'error': f'Valor inválido: {col}'}), 400
            valores.append(valor)

        datos = pd.DataFrame([valores], columns=COLUMNAS)
        
        # Predecir cluster
        entrada = scaler.transform(datos)
        cluster = modelo.predict(entrada)[0]
        gama = mapeo[cluster]
        
        # Obtener promedios del cluster
        cluster_data = promedios[promedios['cluster'] == cluster].iloc[0]
        promedio = {
            'Núcleos': int(cluster_data['num_cores']),
            'Velocidad (GHz)': round(cluster_data['processor_speed'], 1),
            'Batería (mAh)': int(cluster_data['battery_capacity']),
            'Carga rápida': 'Sí' if cluster_data['fast_charging_available'] > 0.5 else 'No',
            'RAM (GB)': int(cluster_data['ram_capacity']),
            'Almacenamiento (GB)': int(cluster_data['internal_memory']),
            'Pantalla (pulg)': round(cluster_data['screen_size'], 1),
            'Resolución': f"{int(cluster_data['resolution_width'])}x{int(cluster_data['resolution_height'])}",
            'Cámaras traseras': int(cluster_data['num_rear_cameras']),
            'Cámara principal (MP)': int(cluster_data['primary_camera_rear']),
            'Cámara frontal (MP)': int(cluster_data['primary_camera_front']),
            'Precio (USD)': f"${int(cluster_data['price'])}"
        }
        
        # Identificar factores clave (diferencias relativas reales)
        factores = calcular_factores_clave(datos.iloc[0].to_dict(), cluster_data)
        
        # Datos para gráficos
        dif_relativas = calcular_dif_relativas(entrada, cluster)
        
        # Reducir a 2D (solo una vez por petición)
        pca = PCA(n_components=2)
        centroides_scaled = modelo.cluster_centers_
        pca.fit(centroides_scaled)

        # Transformar clusters y el usuario al nuevo espacio 2D
        clusters_2d = pca.transform(centroides_scaled)
        usuario_2d = pca.transform(entrada)

        # Calcular distancia del usuario a cada cluster
        distancias = euclidean_distances(usuario_2d, clusters_2d)[0]
        cluster_mas_cercano = int(np.argmin(distancias))
        distancia_minima = round(float(distancias[cluster_mas_cercano]), 3)
        gama_cercana = mapeo[cluster_mas_cercano]
        
        return jsonify({
            'gama': gama,
            'promedio': promedio,
            'factores_clave': factores,
            'dif_relativas': dif_relativas,
            'pca_clusters': clusters_2d.tolist(),
            'pca_usuario': usuario_2d[0].tolist(),
            'gamas': [mapeo[i] for i in range(len(centroides_scaled))],
            'distancia_minima': distancia_minima,
            'gama_cercana': gama_cercana,
            'distribucion': distribucion,
            'total_dispositivos': sum(distribucion.values())
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

def calcular_factores_clave(usuario, cluster_data):
    """Identifica los 3 factores con mayor diferencia relativa"""
    campos = {
        'price': 'precio',
        'ram_capacity': 'memoria RAM',
        'primary_camera_rear': 'cámara principal',
        'processor_speed': 'velocidad del procesador',
        'battery_capacity': 'capacidad de batería',
        'internal_memory': 'almacenamiento interno'
    }
    
    diferencias = []
    for campo, nombre in campos.items():
        val_usuario = usuario[campo]
        val_promedio = cluster_data[campo]
        
        if val_promedio > 0:
            dif_rel = abs(val_usuario - val_promedio) / val_promedio
            diferencias.append((nombre, dif_rel))
    
    diferencias.sort(key=lambda x: x[1], reverse=False)
    return [f[0] for f in diferencias[:5]]

def calcular_dif_relativas(usuario_scaled, cluster):
    """Calcula diferencias relativas y puntajes escalados por categoría"""
    categorias = {
        'Procesador': [0, 1],
        'Memoria': [4, 5],
        'Pantalla': [6, 7, 8],
        'Cámara': [9, 10, 11],
        'Batería': [2, 3],
        'Precio': [12]
    }

    # ---- Cálculo de valores base ----
    cluster_row = promedios[promedios['cluster'] == cluster].iloc[0]
    cluster_valores = cluster_row[COLUMNAS].values.reshape(1, -1)

    # Convertir a DataFrame para cálculos relativos
    df_cluster = pd.DataFrame(cluster_valores, columns=COLUMNAS)
    df_usuario = pd.DataFrame(scaler.inverse_transform(usuario_scaled), columns=COLUMNAS)

    # ---- 1️⃣ Diferencias relativas por categoría ----
    diff_relativas = {}
    for cat, indices in categorias.items():
        difs = []
        for idx in indices:
            col = COLUMNAS[idx]
            v_user = df_usuario.iloc[0][col]
            v_cluster = df_cluster.iloc[0][col]
            if v_cluster != 0:
                difs.append((v_user - v_cluster) / v_cluster)
        diff_relativas[cat] = float(np.mean(difs)) if difs else 0.0

    return diff_relativas

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*50)
    print("Servidor Flask - Clasificador de Celulares")
    print("="*50)
    if not modelos_ok:
        print("⚠️  ADVERTENCIA: Ejecuta 'python model.py' primero")
    print("URL: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)