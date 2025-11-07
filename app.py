from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from io import BytesIO
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'

# Cargar modelos y datos
try:
    scaler = joblib.load('scaler.pkl')
    modelo = joblib.load('modelo_kmeans.pkl')
    mapeo = joblib.load('mapeo_gama.pkl')
    promedios = pd.read_csv('clusters_promedios.csv')
    pca = joblib.load('pca.pkl')
    print("Modelos cargados correctamente")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    # Crear datos de ejemplo si no existen los modelos
    promedios = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/resultado', methods=['POST'])
def resultado_api():
    try:
        data = request.get_json()
        
        # Validar datos requeridos
        required_fields = [
            'num_cores', 'processor_speed', 'battery_capacity',
            'fast_charging_available', 'ram_capacity', 'internal_memory',
            'screen_size', 'resolution_width', 'resolution_height',
            'num_rear_cameras', 'primary_camera_rear', 'primary_camera_front', 'price'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo faltante: {field}'}), 400
        
        # Crear DataFrame con los datos
        datos = pd.DataFrame([[
            int(data['num_cores']),
            float(data['processor_speed']),
            int(data['battery_capacity']),
            int(data['fast_charging_available']),
            int(data['ram_capacity']),
            int(data['internal_memory']),
            float(data['screen_size']),
            int(data['resolution_width']),
            int(data['resolution_height']),
            int(data['num_rear_cameras']),
            int(data['primary_camera_rear']),
            int(data['primary_camera_front']),
            int(data['price'])
        ]], columns=promedios.columns[1:-1])

        # Preprocesar y predecir
        entrada = scaler.transform(datos)
        cluster = modelo.predict(entrada)[0]
        gama = mapeo[cluster]
        
        # Obtener promedios del cluster
        promedio_cluster = promedios.iloc[cluster, 1:-1].to_dict()
        
        # Generar gráfico y obtener factores clave
        grafico_base64 = generar_grafico_personalizado(entrada, cluster)
        factores_clave = identificar_factores_clave(datos.iloc[0].to_dict(), promedio_cluster, gama)
        
        return jsonify({
            'gama': gama,
            'promedio': promedio_cluster,
            'factores_clave': factores_clave,
            'grafico': grafico_base64
        })
        
    except Exception as e:
        print(f"Error en el procesamiento: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

def identificar_factores_clave(datos_usuario, promedio_cluster, gama):
    """Identifica los factores más importantes que determinan la gama"""
    factores = []
    
    # Mapeo de nombres amigables
    nombres_amigables = {
        'num_cores': 'número de núcleos',
        'processor_speed': 'velocidad del procesador',
        'battery_capacity': 'capacidad de la batería',
        'ram_capacity': 'memoria RAM',
        'internal_memory': 'almacenamiento interno',
        'screen_size': 'tamaño de pantalla',
        'resolution_width': 'resolución de pantalla',
        'num_rear_cameras': 'número de cámaras traseras',
        'primary_camera_rear': 'cámara principal trasera',
        'primary_camera_front': 'cámara frontal',
        'price': 'precio'
    }
    
    # Umbrales para considerar como factor clave
    umbrales = {
        'num_cores': 2,
        'processor_speed': 0.5,
        'battery_capacity': 1000,
        'ram_capacity': 2,
        'internal_memory': 32,
        'screen_size': 0.5,
        'resolution_width': 500,
        'num_rear_cameras': 1,
        'primary_camera_rear': 16,
        'primary_camera_front': 8,
        'price': 100
    }
    
    for campo, valor_usuario in datos_usuario.items():
        if campo in promedio_cluster and campo in umbrales:
            valor_promedio = promedio_cluster[campo]
            diferencia = abs(valor_usuario - valor_promedio)
            
            if diferencia >= umbrales[campo]:
                factor = nombres_amigables.get(campo, campo)
                factores.append(factor)
    
    # Si no hay factores significativos, usar los más comunes
    if not factores:
        factores = ['precio', 'memoria RAM', 'cámara principal']
    
    return factores[:3]  # Devolver solo los 3 factores más importantes

def generar_grafico_personalizado(usuario_scaled, cluster_usuario):
    """Genera un gráfico personalizado y lo devuelve en base64"""
    try:
        # Configurar estilo del gráfico
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico 1: Comparación con promedios de gama
        categorias = ['Procesador', 'Memoria', 'Pantalla', 'Cámara', 'Batería', 'Precio']
        usuario_puntajes = [
            (usuario_scaled[0][0] + usuario_scaled[0][1]) / 2,  # Procesador
            (usuario_scaled[0][4] + usuario_scaled[0][5]) / 2,  # Memoria
            (usuario_scaled[0][6] + usuario_scaled[0][7] + usuario_scaled[0][8]) / 3,  # Pantalla
            (usuario_scaled[0][9] + usuario_scaled[0][10] + usuario_scaled[0][11]) / 3,  # Cámara
            (usuario_scaled[0][2] + usuario_scaled[0][3]) / 2,  # Batería
            usuario_scaled[0][12]  # Precio
        ]
        
        # Obtener promedios del cluster para comparar
        cluster_puntajes = [
            (promedios.iloc[cluster_usuario, 1] + promedios.iloc[cluster_usuario, 2]) / 2,
            (promedios.iloc[cluster_usuario, 5] + promedios.iloc[cluster_usuario, 6]) / 2,
            (promedios.iloc[cluster_usuario, 7] + promedios.iloc[cluster_usuario, 8] + promedios.iloc[cluster_usuario, 9]) / 3,
            (promedios.iloc[cluster_usuario, 10] + promedios.iloc[cluster_usuario, 11] + promedios.iloc[cluster_usuario, 12]) / 3,
            (promedios.iloc[cluster_usuario, 3] + promedios.iloc[cluster_usuario, 4]) / 2,
            promedios.iloc[cluster_usuario, 13]
        ]
        
        x = np.arange(len(categorias))
        width = 0.35
        
        ax1.bar(x - width/2, usuario_puntajes, width, label='Tu dispositivo', alpha=0.8, color='#0078d4')
        ax1.bar(x + width/2, cluster_puntajes, width, label=f'Promedio {mapeo[cluster_usuario]}', alpha=0.8, color='#28a745')
        
        ax1.set_xlabel('Categorías')
        ax1.set_ylabel('Puntuación Estandarizada')
        ax1.set_title('Comparación con Promedio de Gama')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categorias, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Distribución de gamas
        gamas = ['Gama Baja', 'Gama Media', 'Gama Alta']
        colores = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        
        # Simular distribución (en un caso real, usarías datos reales)
        distribucion = [40, 35, 25]  # Porcentajes aproximados
        
        wedges, texts, autotexts = ax2.pie(distribucion, labels=gamas, autopct='%1.1f%%',
                                          colors=colores, startangle=90)
        
        # Resaltar la gama del usuario
        gama_usuario_idx = list(mapeo.values()).index(mapeo[cluster_usuario])
        wedges[gama_usuario_idx].set_edgecolor('black')
        wedges[gama_usuario_idx].set_linewidth(2)
        
        ax2.set_title('Distribución de Gamas en el Mercado')
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"Error generando gráfico: {e}")
        # Devolver una imagen placeholder en caso de error
        return generar_grafico_placeholder()

def generar_grafico_placeholder():
    """Genera un gráfico placeholder en caso de error"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Gráfico no disponible\nen modo demostración', 
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=16)
    ax.set_facecolor('#f8f9fa')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

@app.route('/api/ejemplo', methods=['GET'])
def datos_ejemplo():
    """Endpoint para obtener datos de ejemplo para testing"""
    ejemplo = {
        'num_cores': 8,
        'processor_speed': 2.8,
        'battery_capacity': 4500,
        'fast_charging_available': 1,
        'ram_capacity': 8,
        'internal_memory': 128,
        'screen_size': 6.7,
        'resolution_width': 1440,
        'resolution_height': 3120,
        'num_rear_cameras': 3,
        'primary_camera_rear': 64,
        'primary_camera_front': 32,
        'price': 650
    }
    return jsonify(ejemplo)

if __name__ == '__main__':
    # Crear directorio para templates si no existe
    os.makedirs('templates', exist_ok=True)
    
    print("Iniciando servidor Flask...")
    print("URL: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)