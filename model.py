import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from sklearn.decomposition import PCA

# Cargar dataset
df = pd.read_csv('data/celulares.csv')

# Selección de características
features = df[[
    'num_cores', 'processor_speed',
    'battery_capacity', 'fast_charging_available',
    'ram_capacity', 'internal_memory',
    'screen_size', 'resolution_width', 'resolution_height',
    'num_rear_cameras', 'primary_camera_rear', 'primary_camera_front',
    'price'
]]

# Convertir 'fast_charging_available' a entero
features['fast_charging_available'] = features['fast_charging_available'].astype(int)

# Verificar valores faltantes
print("Valores faltantes por columna:")
print(features.isnull().sum())

# Eliminar filas con NaN (alternativa: usar imputación si prefieres conservar más datos)
features = features.dropna()

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Reducir a 2 dimensiones para graficar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Mapeo de clusters a gama
centroides = kmeans.cluster_centers_
suma_centroides = centroides.sum(axis=1)
orden = pd.Series(suma_centroides).rank(method='dense').astype(int) - 1
mapeo = {i: f'Gama {"Baja" if r == 0 else "Media" if r == 1 else "Alta"}' for i, r in enumerate(orden)}

# Guardar modelos
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'modelo_kmeans.pkl')
joblib.dump(mapeo, 'mapeo_gama.pkl')
features['cluster'] = kmeans.predict(X_scaled)
features.groupby('cluster').mean().to_csv('clusters_promedios.csv')

# Guardar PCA y componentes
joblib.dump(pca, 'pca.pkl')
pd.DataFrame(X_pca, columns=['PC1', 'PC2']).to_csv('pca_coords.csv', index=False)

print("Modelo entrenado y guardado.")