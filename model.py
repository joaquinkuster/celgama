"""
Sistema de Clustering de Celulares usando K-Means
==================================================

Este módulo entrena un modelo de clustering K-Means para clasificar celulares
en diferentes gamas (Baja, Media, Alta) basándose en sus características técnicas.

Características principales:
- Preprocesamiento de datos con StandardScaler
- Clustering con K-Means (3 clusters)
- Evaluación con múltiples métricas (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Análisis de balance de clusters
- Mapeo inteligente de clusters a gamas basado en múltiples factores

Autor: Sistema de ML
Fecha: 2025-11-20
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import json
import os
import sys
from typing import Dict, List, Tuple, Any

# ===== CONFIGURACIÓN =====
print("=" * 50)
print("Entrenando modelo de clasificación")
print("=" * 50)

# Constantes de configuración
INR_TO_USD: float = 90.0
K_CLUSTERS: int = 3
RANDOM_STATE: int = 42
BALANCE_THRESHOLD: float = 0.15  # 15% de desbalance máximo aceptable

# Features utilizadas para el clustering
FEATURES: List[str] = [
    'num_cores', 'processor_speed', 'battery_capacity', 'fast_charging_available',
    'ram_capacity', 'internal_memory', 'screen_size', 'resolution_width',
    'resolution_height', 'num_rear_cameras', 'primary_camera_rear',
    'primary_camera_front', 'price'
]

# Pesos para el cálculo del puntaje compuesto
FEATURE_WEIGHTS: Dict[str, float] = {
    'price': 0.25,
    'ram_capacity': 0.20,
    'primary_camera_rear': 0.15,
    'processor_speed': 0.15,
    'battery_capacity': 0.15,
    'internal_memory': 0.10
}


def cargar_dataset(file_path: str) -> pd.DataFrame:
    """
    Carga el dataset de celulares desde un archivo CSV.
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos cargados
        
    Raises:
        SystemExit: Si el archivo no existe o faltan columnas requeridas
    """
    if not os.path.exists(file_path):
        print(f"❌ ERROR: No se encontró '{file_path}'")
        sys.exit(1)
    
    print(f"\n1. Cargando dataset desde '{file_path}'...")
    df = pd.read_csv(file_path)
    print(f"   ✓ {len(df)} registros cargados")
    
    # Verificar que todas las columnas necesarias existan
    if not all(col in df.columns for col in FEATURES):
        missing = [col for col in FEATURES if col not in df.columns]
        print(f"❌ ERROR: Faltan columnas: {', '.join(missing)}")
        sys.exit(1)
    
    return df


def convertir_precios(df: pd.DataFrame, conversion_rate: float) -> pd.DataFrame:
    """
    Convierte los precios de INR a USD.
    
    Args:
        df: DataFrame con los datos
        conversion_rate: Tasa de conversión INR a USD
        
    Returns:
        DataFrame con precios convertidos
    """
    print(f"\n2. Convirtiendo precios (1 USD = {conversion_rate} INR)...")
    df['price'] = df['price'] / conversion_rate
    print(f"   ✓ Rango: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y prepara los datos para el clustering.
    
    Args:
        df: DataFrame con los datos originales
        
    Returns:
        DataFrame limpio con solo las features necesarias
    """
    print("\n3. Limpiando datos...")
    features = df[FEATURES].copy()
    features['fast_charging_available'] = features['fast_charging_available'].astype(int)
    features = features.dropna()
    print(f"   ✓ {len(features)} registros limpios")
    return features


def escalar_datos(features: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Escala los datos usando StandardScaler.
    
    Args:
        features: DataFrame con las características
        
    Returns:
        Tupla con (datos escalados, scaler entrenado)
    """
    print("\n4. Escalando datos...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    print("   ✓ Datos escalados con StandardScaler")
    return X_scaled, scaler


def entrenar_kmeans(X_scaled: np.ndarray, n_clusters: int, random_state: int) -> Tuple[KMeans, np.ndarray]:
    """
    Entrena el modelo K-Means buscando la mejor configuración balanceada.
    
    Prueba múltiples inicializaciones y selecciona la que tenga:
    1. Mejor balance (distribución más uniforme)
    2. Métricas de calidad aceptables
    
    Args:
        X_scaled: Datos escalados
        n_clusters: Número de clusters
        random_state: Semilla base para reproducibilidad
        
    Returns:
        Tupla con (modelo entrenado, etiquetas predichas)
    """
    print(f"\n5. Entrenando K-Means ({n_clusters} clusters) con balanceo optimizado...")
    
    mejor_modelo = None
    mejor_labels = None
    mejor_score = float('inf')  # Score combinado: balance + calidad
    mejor_desbalance = float('inf')
    mejor_silhouette = -1
    
    # Probar múltiples inicializaciones (más intentos para mejor balance)
    n_intentos = 50
    print(f"   Probando {n_intentos} configuraciones diferentes...")
    
    for i in range(n_intentos):
        # Usar diferentes semillas para cada intento
        seed = random_state + i * 7  # Multiplicar para mayor variación
        kmeans_temp = KMeans(n_clusters=n_clusters, random_state=seed, n_init=15, max_iter=400)
        kmeans_temp.fit(X_scaled)
        labels_temp = kmeans_temp.predict(X_scaled)
        
        # Calcular balance
        unique, counts = np.unique(labels_temp, return_counts=True)
        total = len(labels_temp)
        porcentajes = counts / total
        desbalance = max(porcentajes) - min(porcentajes)
        
        # Calcular silhouette para asegurar calidad
        silhouette = silhouette_score(X_scaled, labels_temp)
        
        # Score combinado: priorizar balance (70%) y calidad (30%)
        # Normalizar desbalance a [0, 1] y silhouette a [0, 1]
        score_balance = desbalance  # Menor es mejor
        score_calidad = (1 - silhouette) if silhouette > 0 else 1  # Menor es mejor
        score_combinado = 0.7 * score_balance + 0.3 * score_calidad
        
        # Seleccionar si tiene mejor score Y calidad mínima aceptable
        if score_combinado < mejor_score and silhouette > 0.15:
            mejor_score = score_combinado
            mejor_desbalance = desbalance
            mejor_modelo = kmeans_temp
            mejor_labels = labels_temp
            mejor_silhouette = silhouette
    
    # Si no encontramos ninguno con calidad aceptable, relajar restricción
    if mejor_modelo is None:
        print("   ⚠️  No se encontró configuración con calidad mínima de 0.15")
        print("   Buscando con umbral más bajo...")
        
        for i in range(n_intentos):
            seed = random_state + i * 7
            kmeans_temp = KMeans(n_clusters=n_clusters, random_state=seed, n_init=15, max_iter=400)
            kmeans_temp.fit(X_scaled)
            labels_temp = kmeans_temp.predict(X_scaled)
            
            unique, counts = np.unique(labels_temp, return_counts=True)
            total = len(labels_temp)
            porcentajes = counts / total
            desbalance = max(porcentajes) - min(porcentajes)
            
            silhouette = silhouette_score(X_scaled, labels_temp)
            score_balance = desbalance
            score_calidad = (1 - silhouette) if silhouette > 0 else 1
            score_combinado = 0.7 * score_balance + 0.3 * score_calidad
            
            if score_combinado < mejor_score:
                mejor_score = score_combinado
                mejor_desbalance = desbalance
                mejor_modelo = kmeans_temp
                mejor_labels = labels_temp
                mejor_silhouette = silhouette
    
    print(f"   ✓ Mejor configuración encontrada:")
    print(f"     - Desbalance: {mejor_desbalance*100:.1f}%")
    print(f"     - Silhouette Score: {mejor_silhouette:.4f}")
    print(f"     - Score combinado: {mejor_score:.4f}")
    
    return mejor_modelo, mejor_labels


def calcular_metricas_clustering(X_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de evaluación del clustering.
    
    Métricas calculadas:
    - Silhouette Score: Mide qué tan similar es un objeto a su propio cluster vs otros clusters
      Rango: [-1, 1], valores más altos son mejores (>0.5 excelente, >0.3 aceptable)
    - Davies-Bouldin Index: Mide la separación entre clusters
      Rango: [0, ∞), valores más bajos son mejores (<1.0 excelente, <2.0 aceptable)
    - Calinski-Harabasz Score: Ratio de dispersión entre clusters vs dentro de clusters
      Rango: [0, ∞), valores más altos son mejores (>300 bueno)
    - Inertia: Suma de distancias cuadradas a los centroides
      Valores más bajos son mejores (relativo al dataset)
    
    Args:
        X_scaled: Datos escalados
        labels: Etiquetas de cluster asignadas
        
    Returns:
        Diccionario con las métricas calculadas
    """
    print("\n6. Calculando métricas de evaluación...")
    
    metricas = {}
    
    # Silhouette Score
    silhouette = silhouette_score(X_scaled, labels)
    metricas['silhouette_score'] = float(silhouette)
    print(f"   • Silhouette Score: {silhouette:.4f}")
    
    # Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    metricas['davies_bouldin_index'] = float(davies_bouldin)
    print(f"   • Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    # Calinski-Harabasz Score
    calinski = calinski_harabasz_score(X_scaled, labels)
    metricas['calinski_harabasz_score'] = float(calinski)
    print(f"   • Calinski-Harabasz Score: {calinski:.2f}")
    
    return metricas


def interpretar_metricas(metricas: Dict[str, float]) -> Dict[str, str]:
    """
    Interpreta las métricas de clustering y proporciona evaluación cualitativa.
    
    Args:
        metricas: Diccionario con las métricas calculadas
        
    Returns:
        Diccionario con interpretaciones de cada métrica
    """
    interpretaciones = {}
    
    # Interpretar Silhouette Score
    silhouette = metricas['silhouette_score']
    if silhouette > 0.5:
        interpretaciones['silhouette'] = "Excelente"
    elif silhouette > 0.3:
        interpretaciones['silhouette'] = "Aceptable"
    elif silhouette > 0.0:
        interpretaciones['silhouette'] = "Débil"
    else:
        interpretaciones['silhouette'] = "Pobre"
    
    # Interpretar Davies-Bouldin Index
    davies = metricas['davies_bouldin_index']
    if davies < 1.0:
        interpretaciones['davies_bouldin'] = "Excelente"
    elif davies < 2.0:
        interpretaciones['davies_bouldin'] = "Bueno"
    else:
        interpretaciones['davies_bouldin'] = "Necesita mejora"
    
    # Interpretar Calinski-Harabasz Score
    calinski = metricas['calinski_harabasz_score']
    if calinski > 300:
        interpretaciones['calinski'] = "Bueno"
    elif calinski > 100:
        interpretaciones['calinski'] = "Aceptable"
    else:
        interpretaciones['calinski'] = "Débil"
    
    return interpretaciones


def verificar_balance_clusters(labels: np.ndarray, threshold: float = BALANCE_THRESHOLD) -> Dict[str, Any]:
    """
    Verifica el balance de la distribución de datos entre clusters.
    
    Args:
        labels: Etiquetas de cluster asignadas
        threshold: Umbral de desbalance aceptable (default: 15%)
        
    Returns:
        Diccionario con información sobre el balance
    """
    print("\n7. Verificando balance de clusters...")
    
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    distribuciones = {}
    for cluster_id, count in zip(unique, counts):
        porcentaje = count / total
        distribuciones[int(cluster_id)] = {
            'count': int(count),
            'percentage': float(porcentaje)
        }
        print(f"   • Cluster {cluster_id}: {count} muestras ({porcentaje*100:.1f}%)")
    
    # Calcular desbalance (diferencia entre max y min)
    porcentajes = [d['percentage'] for d in distribuciones.values()]
    desbalance = max(porcentajes) - min(porcentajes)
    
    es_balanceado = desbalance <= threshold
    
    if es_balanceado:
        print(f"   ✓ Clusters balanceados (desbalance: {desbalance*100:.1f}%)")
    else:
        print(f"   ⚠️  Clusters desbalanceados (desbalance: {desbalance*100:.1f}%)")
        print(f"      Considerar técnicas de balanceo si el desbalance es problemático")
    
    return {
        'distribuciones': distribuciones,
        'desbalance': float(desbalance),
        'es_balanceado': es_balanceado,
        'threshold': threshold
    }


def rebalancear_clusters(X_scaled: np.ndarray, kmeans: KMeans, labels: np.ndarray, 
                         target_balance: float = 0.20) -> np.ndarray:
    """
    Re-balancea los clusters redistribuyendo puntos en las fronteras.
    
    Identifica puntos que están cerca de múltiples centroides y los reasigna
    para lograr una distribución más equilibrada sin sacrificar demasiado la calidad.
    
    Args:
        X_scaled: Datos escalados
        kmeans: Modelo K-Means entrenado
        labels: Etiquetas originales
        target_balance: Desbalance objetivo (default: 20%)
        
    Returns:
        Nuevas etiquetas balanceadas
    """
    print("\n7.5. Aplicando re-balanceo de clusters...")
    
    from sklearn.metrics import euclidean_distances
    
    # Calcular distancias a todos los centroides
    centroides = kmeans.cluster_centers_
    distancias = euclidean_distances(X_scaled, centroides)
    
    # Identificar puntos "ambiguos" (cerca de múltiples centroides)
    distancias_ordenadas = np.sort(distancias, axis=1)
    diff_relativa = (distancias_ordenadas[:, 1] - distancias_ordenadas[:, 0]) / distancias_ordenadas[:, 0]
    puntos_ambiguos = diff_relativa < 0.30  # Puntos con <30% diferencia entre 1° y 2° centroide
    
    # Copiar labels originales
    nuevas_labels = labels.copy()
    
    # Calcular distribución actual
    unique, counts = np.unique(nuevas_labels, return_counts=True)
    total = len(nuevas_labels)
    distribuciones = dict(zip(unique, counts))
    
    # Identificar cluster más grande y más pequeño
    cluster_max = max(distribuciones, key=distribuciones.get)
    cluster_min = min(distribuciones, key=distribuciones.get)
    
    desbalance_actual = (max(counts) - min(counts)) / total
    
    print(f"   Desbalance inicial: {desbalance_actual*100:.1f}%")
    print(f"   Cluster más grande: {cluster_max} ({distribuciones[cluster_max]} puntos)")
    print(f"   Cluster más pequeño: {cluster_min} ({distribuciones[cluster_min]} puntos)")
    
    # Reasignar puntos ambiguos del cluster más grande a otros clusters
    puntos_a_mover = 0
    for idx in np.where(puntos_ambiguos)[0]:
        if nuevas_labels[idx] == cluster_max:
            # Encontrar el segundo centroide más cercano
            clusters_ordenados = np.argsort(distancias[idx])
            segundo_cluster = clusters_ordenados[1]
            
            # Mover si ayuda al balance
            if distribuciones[cluster_max] > distribuciones[segundo_cluster]:
                nuevas_labels[idx] = segundo_cluster
                distribuciones[cluster_max] -= 1
                distribuciones[segundo_cluster] += 1
                puntos_a_mover += 1
                
                # Verificar si ya alcanzamos el balance objetivo
                desbalance_nuevo = (max(distribuciones.values()) - min(distribuciones.values())) / total
                if desbalance_nuevo <= target_balance:
                    break
    
    # Calcular nuevo desbalance
    unique_new, counts_new = np.unique(nuevas_labels, return_counts=True)
    desbalance_final = (max(counts_new) - min(counts_new)) / total
    
    print(f"   Puntos reasignados: {puntos_a_mover}")
    print(f"   Desbalance final: {desbalance_final*100:.1f}%")
    
    if desbalance_final < desbalance_actual:
        print(f"   ✓ Mejora: {(desbalance_actual - desbalance_final)*100:.1f}%")
        return nuevas_labels
    else:
        print(f"   ⚠️  No se logró mejorar el balance, manteniendo asignación original")
        return labels


def mapear_clusters_a_gamas(features: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> Dict[int, str]:
    """
    Mapea clusters a gamas basándose en un puntaje compuesto de múltiples características.
    
    Args:
        features: DataFrame con las características
        labels: Etiquetas de cluster
        n_clusters: Número de clusters
        
    Returns:
        Diccionario mapeando cluster_id -> nombre_gama
    """
    print("\n8. Mapeando clusters a gamas (análisis multi-característica)...")
    features_copy = features.copy()
    features_copy['cluster'] = labels
    
    # Calcular puntajes compuestos por cluster
    puntajes_cluster = []
    for cluster_id in range(n_clusters):
        cluster_data = features_copy[features_copy['cluster'] == cluster_id]
        
        # Puntaje basado en múltiples factores con pesos configurables
        puntaje = (
            cluster_data['price'].mean() * FEATURE_WEIGHTS['price'] +
            cluster_data['ram_capacity'].mean() * FEATURE_WEIGHTS['ram_capacity'] +
            cluster_data['primary_camera_rear'].mean() * FEATURE_WEIGHTS['primary_camera_rear'] +
            cluster_data['processor_speed'].mean() * FEATURE_WEIGHTS['processor_speed'] +
            cluster_data['battery_capacity'].mean() / 100 * FEATURE_WEIGHTS['battery_capacity'] +
            cluster_data['internal_memory'].mean() * FEATURE_WEIGHTS['internal_memory']
        )
        puntajes_cluster.append((cluster_id, puntaje))
    
    # Ordenar por puntaje y asignar gamas
    puntajes_cluster.sort(key=lambda x: x[1])
    mapeo = {}
    gamas = ['Gama Baja', 'Gama Media', 'Gama Alta']
    
    for idx, (cluster_id, _) in enumerate(puntajes_cluster):
        mapeo[cluster_id] = gamas[idx]
    
    # Mostrar mapeo
    print("   Mapeo de clusters:")
    for cluster_id, gama in sorted(mapeo.items()):
        count = (labels == cluster_id).sum()
        cluster_data = features_copy[features_copy['cluster'] == cluster_id]
        avg_price = cluster_data['price'].mean()
        avg_ram = cluster_data['ram_capacity'].mean()
        print(f"   - Cluster {cluster_id} → {gama} ({count} dispositivos, ${avg_price:.0f}, {avg_ram:.0f}GB RAM)")
    
    return mapeo


def guardar_modelos(scaler: StandardScaler, kmeans: KMeans, mapeo: Dict[int, str], 
                    distribucion: Dict[str, int], metricas: Dict[str, float], 
                    balance_info: Dict[str, Any]) -> None:
    """
    Guarda todos los modelos y metadatos en archivos.
    
    Args:
        scaler: Escalador entrenado
        kmeans: Modelo K-Means entrenado
        mapeo: Mapeo de clusters a gamas
        distribucion: Distribución de dispositivos por gama
        metricas: Métricas de evaluación
        balance_info: Información sobre balance de clusters
    """
    print("\n9. Guardando modelos y métricas...")
    
    # Guardar modelos pickle
    archivos_pkl = {
        'scaler.pkl': scaler,
        'modelo_kmeans.pkl': kmeans,
        'mapeo_gama.pkl': mapeo,
        'distribucion_gamas.pkl': distribucion
    }
    
    for filename, obj in archivos_pkl.items():
        joblib.dump(obj, filename)
        print(f"   ✓ {filename}")
    
    # Guardar métricas en JSON
    metricas_completas = {
        'metricas_evaluacion': metricas,
        'interpretaciones': interpretar_metricas(metricas),
        'balance': balance_info,
        'configuracion': {
            'n_clusters': K_CLUSTERS,
            'random_state': RANDOM_STATE,
            'features': FEATURES,
            'feature_weights': FEATURE_WEIGHTS
        }
    }
    
    with open('metricas_clustering.json', 'w', encoding='utf-8') as f:
        json.dump(metricas_completas, f, indent=2, ensure_ascii=False)
    print(f"   ✓ metricas_clustering.json")


def mostrar_reporte_final(features: pd.DataFrame, mapeo: Dict[int, str], 
                          distribucion: Dict[str, int], metricas: Dict[str, float]) -> None:
    """
    Muestra un reporte final con estadísticas y evaluación del modelo.
    
    Args:
        features: DataFrame con características y clusters asignados
        mapeo: Mapeo de clusters a gamas
        distribucion: Distribución de dispositivos por gama
        metricas: Métricas de evaluación
    """
    print("\n" + "=" * 50)
    print("✅ Modelo entrenado exitosamente")
    print("=" * 50)
    
    # Estadísticas generales
    print(f"\nTotal dispositivos: {len(features)}")
    print("\nDistribución por gama:")
    for gama, cantidad in sorted(distribucion.items()):
        porcentaje = (cantidad / len(features)) * 100
        print(f"  - {gama}: {cantidad} ({porcentaje:.1f}%)")
    
    # Evaluación de calidad
    print("\n" + "-" * 50)
    print("EVALUACIÓN DE CALIDAD DEL CLUSTERING")
    print("-" * 50)
    interpretaciones = interpretar_metricas(metricas)
    print(f"Silhouette Score: {metricas['silhouette_score']:.4f} ({interpretaciones['silhouette']})")
    print(f"Davies-Bouldin Index: {metricas['davies_bouldin_index']:.4f} ({interpretaciones['davies_bouldin']})")
    print(f"Calinski-Harabasz Score: {metricas['calinski_harabasz_score']:.2f} ({interpretaciones['calinski']})")
    
    # Promedios por gama
    print("\n" + "-" * 50)
    print("PROMEDIOS POR GAMA")
    print("-" * 50)
    promedios = features.groupby('cluster').mean(numeric_only=True).reset_index()
    
    for cluster_id in sorted(mapeo.keys()):
        gama = mapeo[cluster_id]
        cd = promedios[promedios['cluster'] == cluster_id].iloc[0]
        print(f"\n{gama}:")
        print(f"  - Procesador: {cd['num_cores']:.0f} núcleos @ {cd['processor_speed']:.1f} GHz")
        print(f"  - RAM: {cd['ram_capacity']:.0f} GB | Almacenamiento: {cd['internal_memory']:.0f} GB")
        print(f"  - Batería: {cd['battery_capacity']:.0f} mAh | Cámara: {cd['primary_camera_rear']:.0f} MP")
        print(f"  - Precio: ${cd['price']:.0f} USD")
    
    print("\n" + "=" * 50)
    print("Archivos generados:")
    print("  - scaler.pkl, modelo_kmeans.pkl, mapeo_gama.pkl")
    print("  - distribucion_gamas.pkl, clusters_promedios.csv")
    print("  - metricas_clustering.json")
    print("\nEjecuta ahora: python app.py")
    print("=" * 50 + "\n")


def main() -> None:
    """
    Función principal que ejecuta todo el pipeline de entrenamiento.
    """
    # 1. Cargar dataset
    FILE_PATH = 'data/celulares.csv'
    df = cargar_dataset(FILE_PATH)
    
    # 2. Convertir precios
    df = convertir_precios(df, INR_TO_USD)
    
    # 3. Limpiar datos
    features = limpiar_datos(df)
    
    # 4. Escalar datos
    X_scaled, scaler = escalar_datos(features)
    
    # 5. Entrenar K-Means
    kmeans, labels = entrenar_kmeans(X_scaled, K_CLUSTERS, RANDOM_STATE)
    
    # 6. Calcular métricas de evaluación
    metricas = calcular_metricas_clustering(X_scaled, labels)
    
    # 7. Verificar balance de clusters
    balance_info = verificar_balance_clusters(labels)
    
    # 7.5. Re-balancear clusters si es necesario
    if balance_info['es_balanceado']:  
        labels = rebalancear_clusters(X_scaled, kmeans, labels, target_balance=0.20) # Re-balanceo con objetivo 20%
        # Recalcular balance después del re-balanceo
        balance_info = verificar_balance_clusters(labels)
    
    # 8. Mapear clusters a gamas
    features['cluster'] = labels
    mapeo = mapear_clusters_a_gamas(features, labels, K_CLUSTERS)
    
    # 9. Calcular distribución y estadísticas
    features['gama'] = features['cluster'].map(mapeo)
    distribucion = features.groupby('gama').size().to_dict()
    promedios = features.groupby('cluster').mean(numeric_only=True).reset_index()
    
    # 10. Guardar todo
    promedios.to_csv('clusters_promedios.csv', index=False)
    print("   ✓ clusters_promedios.csv")
    
    guardar_modelos(scaler, kmeans, mapeo, distribucion, metricas, balance_info)
    
    # 11. Mostrar reporte final
    mostrar_reporte_final(features, mapeo, distribucion, metricas)


if __name__ == '__main__':
    main()