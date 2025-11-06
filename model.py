import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Cargar datos
df = pd.read_csv('data/train.csv')

# Selección de características relevantes
features = df[['ram', 'battery_power', 'px_height', 'px_width', 'int_memory']]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Mapeo de clusters a gama según suma de centroides
centroides = kmeans.cluster_centers_
suma_centroides = centroides.sum(axis=1)
orden_gamas = pd.Series(suma_centroides).rank(method='dense').astype(int) - 1
mapeo_gama = {i: f'Gama {"Baja" if r == 0 else "Media" if r == 1 else "Alta"}' for i, r in enumerate(orden_gamas)}

# Guardar modelo, scaler y mapeo
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'modelo_kmeans.pkl')
joblib.dump(mapeo_gama, 'mapeo_gama.pkl')

print("Modelo entrenado y guardado.")