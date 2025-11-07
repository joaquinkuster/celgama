import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from sklearn.decomposition import PCA
import os

print("="*50)
print("Entrenando modelo de clasificación de celulares")
print("="*50)

# Verificar que existe el archivo de datos
if not os.path.exists('data/celulares.csv'):
    print("\n❌ ERROR: No se encontró el archivo 'data/celulares.csv'")
    print("   Por favor, asegúrate de que el archivo existe en la carpeta 'data/'")
    exit(1)

# Cargar dataset
print("\n1. Cargando dataset...")
df = pd.read_csv('data/celulares.csv')
print(f"   ✓ Dataset cargado: {len(df)} registros")

# Convertir precios de INR a USD (1 USD = ~83 INR aproximadamente)
INR_TO_USD = 83.0
print(f"\n2. Convirtiendo precios de INR a USD (1 USD = {INR_TO_USD} INR)...")
if 'price' in df.columns:
    df['price'] = df['price'] / INR_TO_USD
    print(f"   ✓ Precios convertidos")
    print(f"   - Rango de precios: ${df['price'].min():.2f} - ${df['price'].max():.2f} USD")

# Selección de características
print("\n3. Seleccionando características...")
features = df[[
    'num_cores', 'processor_speed',
    'battery_capacity', 'fast_charging_available',
    'ram_capacity', 'internal_memory',
    'screen_size', 'resolution_width', 'resolution_height',
    'num_rear_cameras', 'primary_camera_rear', 'primary_camera_front',
    'price'
]].copy()

# Convertir 'fast_charging_available' a entero
features['fast_charging_available'] = features['fast_charging_available'].astype(int)

# Verificar valores faltantes
print("\n4. Verificando valores faltantes...")
missing_values = features.isnull().sum()
if missing_values.sum() > 0:
    print("   Valores faltantes por columna:")
    print(missing_values[missing_values > 0])
    print(f"\n   ⚠️  Eliminando {features.isnull().any(axis=1).sum()} filas con valores faltantes...")
    features = features.dropna()
else:
    print("   ✓ No hay valores faltantes")

print(f"   ✓ Dataset limpio: {len(features)} registros")

# Escalado
print("\n5. Escalando datos...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
print("   ✓ Datos escalados")

# Reducir a 2 dimensiones para graficar
print("\n6. Aplicando PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
variance_explained = pca.explained_variance_ratio_
print(f"   ✓ PCA aplicado")
print(f"   - Varianza explicada: {variance_explained[0]:.2%} y {variance_explained[1]:.2%}")

# Clustering
print("\n7. Entrenando modelo K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.predict(X_scaled)
print(f"   ✓ Modelo K-Means entrenado con {kmeans.n_clusters} clusters")

# Mapeo de clusters a gama basado en precio promedio
print("\n8. Mapeando clusters a gamas...")
features_copy = features.copy()
features_copy['cluster'] = labels

# Calcular precio promedio por cluster
precio_promedio = features_copy.groupby('cluster')['price'].mean().sort_values()
mapeo = {}
for idx, cluster in enumerate(precio_promedio.index):
    if idx == 0:
        mapeo[cluster] = 'Gama Baja'
    elif idx == 1:
        mapeo[cluster] = 'Gama Media'
    else:
        mapeo[cluster] = 'Gama Alta'

print("   Mapeo de clusters:")
for cluster, gama in sorted(mapeo.items()):
    count = (labels == cluster).sum()
    avg_price = features_copy[features_copy['cluster'] == cluster]['price'].mean()
    print(f"   - Cluster {cluster} → {gama} ({count} dispositivos, ${avg_price:.2f} promedio)")

# Calcular promedios por cluster
print("\n9. Calculando promedios por cluster...")
features_copy['gama'] = features_copy['cluster'].map(mapeo)
promedios = features_copy.groupby('cluster').mean()
promedios['gama'] = promedios.index.map(mapeo)
promedios = promedios.reset_index()

# Contar dispositivos por gama
distribucion = features_copy.groupby('gama').size().to_dict()

# Guardar modelos y datos
print("\n10. Guardando modelos...")
try:
    joblib.dump(scaler, 'scaler.pkl')
    print("   ✓ scaler.pkl guardado")
    
    joblib.dump(kmeans, 'modelo_kmeans.pkl')
    print("   ✓ modelo_kmeans.pkl guardado")
    
    joblib.dump(mapeo, 'mapeo_gama.pkl')
    print("   ✓ mapeo_gama.pkl guardado")
    
    joblib.dump(pca, 'pca.pkl')
    print("   ✓ pca.pkl guardado")
    
    promedios.to_csv('clusters_promedios.csv', index=False)
    print("   ✓ clusters_promedios.csv guardado")
    
    # Guardar coordenadas PCA
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['cluster'] = labels
    pca_df['gama'] = pca_df['cluster'].map(mapeo)
    pca_df.to_csv('pca_coords.csv', index=False)
    print("   ✓ pca_coords.csv guardado")
    
    # Guardar distribución
    joblib.dump(distribucion, 'distribucion_gamas.pkl')
    print("   ✓ distribucion_gamas.pkl guardado")
    
    # Guardar datos completos del dataset procesado
    features_copy.to_csv('dataset_procesado.csv', index=False)
    print("   ✓ dataset_procesado.csv guardado")
    
except Exception as e:
    print(f"   ❌ Error guardando archivos: {e}")
    exit(1)

# Mostrar estadísticas finales
print("\n" + "="*50)
print("✅ Modelo entrenado y guardado exitosamente")
print("="*50)
print(f"\nTotal de dispositivos: {len(features_copy)}")
print("\nDistribución por gama:")
for gama, cantidad in sorted(distribucion.items()):
    porcentaje = (cantidad / len(features_copy)) * 100
    print(f"  - {gama}: {cantidad} dispositivos ({porcentaje:.1f}%)")

print("\nEstadísticas por gama:")
print("-"*50)

for cluster in sorted(mapeo.keys()):
    gama = mapeo[cluster]
    cluster_data = promedios[promedios['cluster'] == cluster].iloc[0]
    
    print(f"\n{gama}:")
    print(f"  - Procesador: {cluster_data['num_cores']:.0f} núcleos @ {cluster_data['processor_speed']:.1f} GHz")
    print(f"  - RAM: {cluster_data['ram_capacity']:.0f} GB")
    print(f"  - Almacenamiento: {cluster_data['internal_memory']:.0f} GB")
    print(f"  - Batería: {cluster_data['battery_capacity']:.0f} mAh")
    print(f"  - Cámara principal: {cluster_data['primary_camera_rear']:.0f} MP")
    print(f"  - Precio promedio: ${cluster_data['price']:.0f} USD")

print("\n" + "="*50)
print("Ahora puedes ejecutar: python app.py")
print("="*50 + "\n")