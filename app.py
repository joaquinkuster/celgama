from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from io import BytesIO
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'

# Cargar modelos y datos
modelos_cargados = False
try:
    scaler = joblib.load('scaler.pkl')
    modelo = joblib.load('modelo_kmeans.pkl')
    mapeo = joblib.load('mapeo_gama.pkl')
    promedios = pd.read_csv('clusters_promedios.csv')
    try:
        pca = joblib.load('pca.pkl')
    except:
        pca = None
        print("PCA no disponible, se generarán gráficos simplificados")
    
    modelos_cargados = True
    print("Modelos cargados correctamente")
    print(f"Columnas de promedios: {promedios.columns.tolist()}")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    print("IMPORTANTE: Ejecuta 'python model.py' primero para generar los modelos")
    # Crear datos de ejemplo si no existen los modelos
    promedios = pd.DataFrame({
        'cluster': [0, 1, 2],
        'num_cores': [4, 6, 8],
        'processor_speed': [1.8, 2.2, 2.8],
        'battery_capacity': [3000, 4000, 5000],
        'fast_charging_available': [0, 1, 1],
        'ram_capacity': [3, 6, 8],
        'internal_memory': [64, 128, 256],
        'screen_size': [5.5, 6.1, 6.7],
        'resolution_width': [720, 1080, 1440],
        'resolution_height': [1520, 2340, 3120],
        'num_rear_cameras': [1, 3, 4],
        'primary_camera_rear': [13, 48, 108],
        'primary_camera_front': [8, 16, 32],
        'price': [150, 350, 800],
        'gama': ['Gama Baja', 'Gama Media', 'Gama Alta']
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/resultado', methods=['POST'])
def resultado_api():
    try:
        # Obtener datos JSON del request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
        
        print(f"Datos recibidos: {data}")
        
        # Validar y convertir datos requeridos
        required_fields = [
            'num_cores', 'processor_speed', 'battery_capacity',
            'ram_capacity', 'internal_memory',
            'screen_size', 'resolution_width', 'resolution_height',
            'num_rear_cameras', 'primary_camera_rear', 'primary_camera_front', 'price'
        ]
        
        for field in required_fields:
            if field not in data or data[field] == '':
                return jsonify({'error': f'Campo faltante o vacío: {field}'}), 400
        
        # Convertir fast_charging_available (checkbox)
        fast_charging = 1 if data.get('fast_charging_available') == '1' or data.get('fast_charging_available') == 'on' else 0
        
        # Si no hay modelos cargados, devolver datos de ejemplo
        if not modelos_cargados:
            return jsonify({
                'gama': 'Gama Media',
                'promedio': {
                    'Núcleos': 6,
                    'Velocidad (GHz)': 2.2,
                    'Batería (mAh)': 4000,
                    'Carga rápida': 'Sí',
                    'RAM (GB)': 6,
                    'Almacenamiento (GB)': 128,
                    'Pantalla (pulg)': 6.1,
                    'Resolución': '1080x2340',
                    'Cámaras traseras': 3,
                    'Cámara principal (MP)': 48,
                    'Cámara frontal (MP)': 16,
                    'Precio (USD)': 350
                },
                'factores_clave': ['precio', 'memoria RAM', 'cámara principal'],
                'grafico': generar_grafico_placeholder()
            })
        
        # Crear DataFrame con los datos en el orden correcto
        columnas_modelo = ['num_cores', 'processor_speed', 'battery_capacity', 
                          'fast_charging_available', 'ram_capacity', 'internal_memory',
                          'screen_size', 'resolution_width', 'resolution_height',
                          'num_rear_cameras', 'primary_camera_rear', 
                          'primary_camera_front', 'price']
        
        datos = pd.DataFrame([[
            float(data['num_cores']),
            float(data['processor_speed']),
            float(data['battery_capacity']),
            float(fast_charging),
            float(data['ram_capacity']),
            float(data['internal_memory']),
            float(data['screen_size']),
            float(data['resolution_width']),
            float(data['resolution_height']),
            float(data['num_rear_cameras']),
            float(data['primary_camera_rear']),
            float(data['primary_camera_front']),
            float(data['price'])
        ]], columns=columnas_modelo)
        
        print(f"DataFrame creado: {datos.values}")

        # Preprocesar y predecir
        entrada = scaler.transform(datos)
        cluster = modelo.predict(entrada)[0]
        gama = mapeo[cluster]
        
        print(f"Cluster predicho: {cluster}, Gama: {gama}")
        
        # Obtener promedios del cluster y formatear
        promedio_cluster = obtener_promedios_formateados(cluster)
        
        # Generar gráfico y obtener factores clave
        grafico_base64 = generar_grafico_personalizado(entrada, cluster, datos)
        factores_clave = identificar_factores_clave(datos.iloc[0].to_dict(), cluster, gama)
        
        return jsonify({
            'gama': gama,
            'promedio': promedio_cluster,
            'factores_clave': factores_clave,
            'grafico': grafico_base64
        })
        
    except Exception as e:
        print(f"Error en el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

def obtener_promedios_formateados(cluster):
    """Obtiene y formatea los promedios del cluster"""
    try:
        cluster_data = promedios[promedios['cluster'] == cluster].iloc[0]
        
        return {
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
            'Precio (USD)': int(cluster_data['price'])
        }
    except Exception as e:
        print(f"Error obteniendo promedios: {e}")
        return {}

def identificar_factores_clave(datos_usuario, cluster, gama):
    """Identifica los factores más importantes que determinan la gama"""
    factores = []
    
    try:
        cluster_data = promedios[promedios['cluster'] == cluster].iloc[0]
        
        # Calcular diferencias relativas
        diferencias = []
        
        campos_importantes = {
            'price': ('precio', 0.2),
            'ram_capacity': ('memoria RAM', 0.3),
            'primary_camera_rear': ('cámara principal', 0.25),
            'processor_speed': ('velocidad del procesador', 0.3),
            'battery_capacity': ('capacidad de batería', 0.15),
            'internal_memory': ('almacenamiento interno', 0.2)
        }
        
        for campo, (nombre, umbral) in campos_importantes.items():
            valor_usuario = datos_usuario[campo]
            valor_promedio = cluster_data[campo]
            
            if valor_promedio > 0:
                dif_relativa = abs(valor_usuario - valor_promedio) / valor_promedio
                if dif_relativa > umbral:
                    diferencias.append((nombre, dif_relativa))
        
        # Ordenar por diferencia y tomar los top 3
        diferencias.sort(key=lambda x: x[1], reverse=True)
        factores = [f[0] for f in diferencias[:3]]
        
    except Exception as e:
        print(f"Error identificando factores: {e}")
    
    # Si no hay factores o hubo error, usar defaults
    if not factores:
        if 'Alta' in gama:
            factores = ['precio', 'cámara principal', 'memoria RAM']
        elif 'Media' in gama:
            factores = ['memoria RAM', 'batería', 'procesador']
        else:
            factores = ['precio', 'pantalla', 'almacenamiento']
    
    return factores[:3]

def generar_grafico_personalizado(usuario_scaled, cluster_usuario, datos_originales):
    """Genera un gráfico personalizado y lo devuelve en base64"""
    try:
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 6))
        
        # Gráfico 1: Comparación con promedios de gama
        ax1 = plt.subplot(1, 2, 1)
        
        categorias = ['Procesador', 'Memoria', 'Pantalla', 'Cámara', 'Batería', 'Precio']
        
        # Calcular puntajes del usuario (normalizados)
        usuario_puntajes = [
            (usuario_scaled[0][0] + usuario_scaled[0][1]) / 2,  # Procesador
            (usuario_scaled[0][4] + usuario_scaled[0][5]) / 2,  # Memoria
            (usuario_scaled[0][6] + usuario_scaled[0][7] + usuario_scaled[0][8]) / 3,  # Pantalla
            (usuario_scaled[0][9] + usuario_scaled[0][10] + usuario_scaled[0][11]) / 3,  # Cámara
            (usuario_scaled[0][2] + usuario_scaled[0][3]) / 2,  # Batería
            usuario_scaled[0][12]  # Precio
        ]
        
        # Obtener promedios del cluster
        cluster_row = promedios[promedios['cluster'] == cluster_usuario].iloc[0]
        
        # Escalar promedios del cluster
        cluster_datos = scaler.transform([[
            cluster_row['num_cores'], cluster_row['processor_speed'],
            cluster_row['battery_capacity'], cluster_row['fast_charging_available'],
            cluster_row['ram_capacity'], cluster_row['internal_memory'],
            cluster_row['screen_size'], cluster_row['resolution_width'],
            cluster_row['resolution_height'], cluster_row['num_rear_cameras'],
            cluster_row['primary_camera_rear'], cluster_row['primary_camera_front'],
            cluster_row['price']
        ]])
        
        cluster_puntajes = [
            (cluster_datos[0][0] + cluster_datos[0][1]) / 2,
            (cluster_datos[0][4] + cluster_datos[0][5]) / 2,
            (cluster_datos[0][6] + cluster_datos[0][7] + cluster_datos[0][8]) / 3,
            (cluster_datos[0][9] + cluster_datos[0][10] + cluster_datos[0][11]) / 3,
            (cluster_datos[0][2] + cluster_datos[0][3]) / 2,
            cluster_datos[0][12]
        ]
        
        x = np.arange(len(categorias))
        width = 0.35
        
        ax1.bar(x - width/2, usuario_puntajes, width, label='Tu dispositivo', 
                alpha=0.8, color='#0078d4')
        ax1.bar(x + width/2, cluster_puntajes, width, 
                label=f'Promedio {mapeo[cluster_usuario]}', alpha=0.8, color='#28a745')
        
        ax1.set_xlabel('Categorías', fontsize=11)
        ax1.set_ylabel('Puntuación Normalizada', fontsize=11)
        ax1.set_title('Comparación con Promedio de Gama', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categorias, rotation=45, ha='right')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Distribución de gamas
        ax2 = plt.subplot(1, 2, 2)
        
        gamas = ['Gama Baja', 'Gama Media', 'Gama Alta']
        colores = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        
        # Contar dispositivos por cluster si hay datos
        if 'cluster' in promedios.columns or len(promedios) == 3:
            distribucion = [33, 34, 33]  # Distribución aproximada
        else:
            distribucion = [40, 35, 25]
        
        wedges, texts, autotexts = ax2.pie(distribucion, labels=gamas, autopct='%1.1f%%',
                                          colors=colores, startangle=90, textprops={'fontsize': 10})
        
        # Resaltar la gama del usuario
        gama_usuario = mapeo[cluster_usuario]
        gama_usuario_idx = gamas.index(gama_usuario) if gama_usuario in gamas else 1
        wedges[gama_usuario_idx].set_edgecolor('black')
        wedges[gama_usuario_idx].set_linewidth(3)
        
        ax2.set_title('Distribución de Gamas\n(Tu gama está resaltada)', 
                     fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"Error generando gráfico: {e}")
        import traceback
        traceback.print_exc()
        return generar_grafico_placeholder()

def generar_grafico_placeholder():
    """Genera un gráfico placeholder en caso de error"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Gráfico de análisis\n(Modo demostración)', 
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=18, color='#6c757d')
    ax.set_facecolor('#f8f9fa')
    ax.axis('off')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

@app.route('/api/ejemplo', methods=['GET'])
def datos_ejemplo():
    """Endpoint para obtener datos de ejemplo para testing"""
    ejemplo = {
        'num_cores': '8',
        'processor_speed': '2.8',
        'battery_capacity': '4500',
        'fast_charging_available': '1',
        'ram_capacity': '8',
        'internal_memory': '128',
        'screen_size': '6.7',
        'resolution_width': '1440',
        'resolution_height': '3120',
        'num_rear_cameras': '3',
        'primary_camera_rear': '64',
        'primary_camera_front': '32',
        'price': '650'
    }
    return jsonify(ejemplo)

if __name__ == '__main__':
    # Crear directorio para templates si no existe
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "="*50)
    print("Iniciando servidor Flask...")
    print("="*50)
    
    if not modelos_cargados:
        print("\n⚠️  ADVERTENCIA: Los modelos no están cargados")
        print("   Ejecuta 'python model.py' primero para entrenar el modelo")
        print("   La aplicación funcionará en modo demostración\n")
    
    print("URL: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)