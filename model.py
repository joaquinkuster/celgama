import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os
import sys

print("="*50)
print("Entrenando modelo de clasificación")
print("="*50)

INR_TO_USD = 90
K_CLUSTERS = 3
RANDOM_STATE = 42

FEATURES = [
    'num_cores', 'processor_speed', 'battery_capacity', 'fast_charging_available',
    'ram_capacity', 'internal_memory', 'screen_size', 'resolution_width',
    'resolution_height', 'num_rear_cameras', 'primary_camera_rear',
    'primary_camera_front', 'price'
]

# 1. Cargar dataset
FILE_PATH = 'data/celulares.csv'
if not os.path.exists(FILE_PATH):
    print(f"❌ ERROR: No se encontró '{FILE_PATH}'")
    sys.exit(1)

print("\n1. Cargando dataset...")
df = pd.read_csv(FILE_PATH)
print(f"   ✓ {len(df)} registros cargados")

# Verificar columnas
if not all(col in df.columns for col in FEATURES):
    missing = [col for col in FEATURES if col not in df.columns]
    print(f"❌ ERROR: Faltan columnas: {', '.join(missing)}")
    sys.exit(1)

# 2. Conversión de precios
print(f"\n2. Convirtiendo precios (1 USD = {INR_TO_USD} INR)...")
df['price'] = df['price'] / INR_TO_USD
print(f"   ✓ Rango: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

# 3. Limpiar datos
print("\n3. Limpiando datos...")
features = df[FEATURES].copy()
features['fast_charging_available'] = features['fast_charging_available'].astype(int)
features = features.dropna()
print(f"   ✓ {len(features)} registros limpios")

# 4. Escalar
print("\n4. Escalando datos...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
print("   ✓ Datos escalados")

# 5. K-Means
print(f"\n5. Entrenando K-Means ({K_CLUSTERS} clusters)...")
kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE, n_init='auto')
kmeans.fit(X_scaled)
labels = kmeans.predict(X_scaled)
print("   ✓ Modelo entrenado")

# 6. Mapeo inteligente (múltiples características)
print("\n6. Mapeando clusters a gamas (análisis multi-característica)...")
features['cluster'] = labels

# Calcular puntajes compuestos por cluster
puntajes_cluster = []
for cluster_id in range(K_CLUSTERS):
    cluster_data = features[features['cluster'] == cluster_id]
    
    # Puntaje basado en múltiples factores con pesos
    puntaje = (
        cluster_data['price'].mean() * 0.25 +
        cluster_data['ram_capacity'].mean() * 0.20 +
        cluster_data['primary_camera_rear'].mean() * 0.15 +
        cluster_data['processor_speed'].mean() * 0.15 +
        cluster_data['battery_capacity'].mean() / 100 * 0.15 +
        cluster_data['internal_memory'].mean() * 0.10
    )
    puntajes_cluster.append((cluster_id, puntaje))

# Ordenar por puntaje y asignar gamas
puntajes_cluster.sort(key=lambda x: x[1])
mapeo = {}
gamas = ['Gama Baja', 'Gama Media', 'Gama Alta']

for idx, (cluster_id, _) in enumerate(puntajes_cluster):
    mapeo[cluster_id] = gamas[idx]

print("   Mapeo de clusters:")
for cluster_id, gama in sorted(mapeo.items()):
    count = (labels == cluster_id).sum()
    avg_price = features[features['cluster'] == cluster_id]['price'].mean()
    avg_ram = features[features['cluster'] == cluster_id]['ram_capacity'].mean()
    print(f"   - Cluster {cluster_id} → {gama} ({count} dispositivos, ${avg_price:.0f}, {avg_ram:.0f}GB RAM)")

# Calcular estadísticas
features['gama'] = features['cluster'].map(mapeo)
promedios = features.groupby('cluster').mean(numeric_only=True).reset_index()
distribucion = features.groupby('gama').size().to_dict()

# 7. Guardar modelos (solo lo esencial)
print("\n7. Guardando modelos...")
archivos = {
    'scaler.pkl': scaler,
    'modelo_kmeans.pkl': kmeans,
    'mapeo_gama.pkl': mapeo,
    'distribucion_gamas.pkl': distribucion
}

for filename, obj in archivos.items():
    joblib.dump(obj, filename)
    print(f"   ✓ {filename}")

promedios.to_csv('clusters_promedios.csv', index=False)
print("   ✓ clusters_promedios.csv")

# 8. Estadísticas finales
print("\n" + "="*50)
print("✅ Modelo entrenado exitosamente")
print("="*50)
print(f"\nTotal dispositivos: {len(features)}")
print("\nDistribución por gama:")
for gama, cantidad in sorted(distribucion.items()):
    porcentaje = (cantidad / len(features)) * 100
    print(f"  - {gama}: {cantidad} ({porcentaje:.1f}%)")

print("\nPromedios por gama:")
print("-" * 50)
for cluster_id in sorted(mapeo.keys()):
    gama = mapeo[cluster_id]
    cd = promedios[promedios['cluster'] == cluster_id].iloc[0]
    print(f"\n{gama}:")
    print(f"  - Procesador: {cd['num_cores']:.0f} núcleos @ {cd['processor_speed']:.1f} GHz")
    print(f"  - RAM: {cd['ram_capacity']:.0f} GB | Almacenamiento: {cd['internal_memory']:.0f} GB")
    print(f"  - Batería: {cd['battery_capacity']:.0f} mAh | Cámara: {cd['primary_camera_rear']:.0f} MP")
    print(f"  - Precio: ${cd['price']:.0f} USD")

print("\n" + "="*50)
print("Ejecuta ahora: python app.py")
print("="*50 + "\n")