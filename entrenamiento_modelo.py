"""
Sistema de Predicción Mundial 2026 - Módulo de Modelado y Entrenamiento
Autor: Científico de Datos / Machine Learning Engineer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import joblib
import os

# ============================================================
# MÓDULO 1: CARGA Y CONSTRUCCIÓN DEL DATASET DE PARTIDOS
# ============================================================


def cargar_y_construir_partidos(ruta_equipos_csv):
    """
    Carga la ficha técnica de los equipos y genera un historial de partidos
    cruzando las estadísticas correspondientes para local y visitante.
    """
    if not os.path.exists(ruta_equipos_csv):
        raise FileNotFoundError(
            f"[ERROR] No se encontró el archivo '{ruta_equipos_csv}'. Primero debés correr 'scraping_y_datos.py'."
        )

    # 1. Cargar las estadísticas base raspadas del script anterior
    df_equipos = pd.read_csv(ruta_equipos_csv)
    lista_equipos = df_equipos['equipo'].tolist()

    print(
        f"[DATA] Ficha técnica de equipos cargada. Selecciones disponibles: {len(lista_equipos)}")

    # 2. GENERACIÓN DEL HISTORIAL DE PARTIDOS (Entorno de desarrollo integrado)
    # Creamos un fixture aleatorio de partidos de entrenamiento usando los equipos reales
    np.random.seed(42)
    n_partidos = 600

    locales = np.random.choice(lista_equipos, size=n_partidos)
    visitantes = np.random.choice(lista_equipos, size=n_partidos)

    # Evitamos que un equipo juegue contra sí mismo en el historial simulado
    for i in range(n_partidos):
        while locales[i] == visitantes[i]:
            visitantes[i] = np.random.choice(lista_equipos)

    df_partidos_historial = pd.DataFrame({
        'local': locales,
        'visitante': visitantes
    })

    # 3. EL TRUCO DEL MERGE: Acoplamos estadísticas del equipo LOCAL
    df_partidos = pd.merge(df_partidos_historial, df_equipos,
                           left_on='local', right_on='equipo', how='left')
    df_partidos = df_partidos.rename(columns={
        'posicion_fifa': 'ranking_fifa_local',
        'promedio_goles_favor_u10': 'promedio_goles_favor_u10_local',
        'promedio_goles_contra_u10': 'promedio_goles_contra_u10_local'
    }).drop(columns=['equipo'])

    # 4. EL TRUCO DEL MERGE 2: Acoplamos estadísticas del equipo VISITANTE
    df_partidos = pd.merge(df_partidos, df_equipos,
                           left_on='visitante', right_on='equipo', how='left')
    df_partidos = df_partidos.rename(columns={
        'posicion_fifa': 'ranking_fifa_visitante',
        'promedio_goles_favor_u10': 'promedio_goles_favor_u10_visitante',
        'promedio_goles_contra_u10': 'promedio_goles_contra_u10_visitante'
    }).drop(columns=['equipo'])

    # Rellenamos nulos por prevención estructural
    df_partidos = df_partidos.fillna({
        'ranking_fifa_local': 99, 'ranking_fifa_visitante': 99,
        'promedio_goles_favor_u10_local': 1.0, 'promedio_goles_favor_u10_visitante': 1.0,
        'promedio_goles_contra_u10_local': 1.0, 'promedio_goles_contra_u10_visitante': 1.0
    })

    # 5. Generación Matemática Coherente de Goles Reales (Targets)
    # Usamos Poisson basado en las métricas verdaderas de ataque y defensa cruzadas
    lambda_local = df_partidos['promedio_goles_favor_u10_local'] * \
        (df_partidos['promedio_goles_contra_u10_visitante'] / 1.5)
    lambda_visitante = df_partidos['promedio_goles_favor_u10_visitante'] * (
        df_partidos['promedio_goles_contra_u10_local'] / 1.5)

    df_partidos['goles_local'] = np.random.poisson(
        lam=np.clip(lambda_local, 0.2, None))
    df_partidos['goles_visitante'] = np.random.poisson(
        lam=np.clip(lambda_visitante, 0.2, None))

    print(
        f"[DATA] Dataset de entrenamiento consolidado. Registros: {df_partidos.shape[0]}, Columnas: {df_partidos.shape[1]}")
    return df_partidos

# ============================================================
# MÓDULO 2: INGENIERÍA DE TARGETS (MÉTRICA 1X2)
# ============================================================


def definir_resultado_1x2(row):
    """Define el output clásico de apuestas: 1 (Gana Local), X (Empate), 2 (Gana Visitante)"""
    if row['goles_local'] > row['goles_visitante']:
        return '1'
    elif row['goles_local'] == row['goles_visitante']:
        return 'X'
    else:
        return '2'

# ============================================================
# MÓDULO 3: PIPELINE DE ENTRENAMIENTO (POISSON ENSEMBLE)
# ============================================================


def entrenar_sistema_predictivo(df):
    # 1. Definir Features (X) y Targets (y)
    features = [
        'ranking_fifa_local', 'ranking_fifa_visitante',
        'promedio_goles_favor_u10_local', 'promedio_goles_favor_u10_visitante',
        'promedio_goles_contra_u10_local', 'promedio_goles_contra_u10_visitante'
    ]

    X = df[features]
    y_local = df['goles_local']
    y_visitante = df['goles_visitante']

    # Calcular etiquetas 1X2 reales para la evaluación final
    y_1x2_real = df.apply(definir_resultado_1x2, axis=1)

    # 2. Train/Test Split (80% entrenamiento, 20% validación interna)
    indices = np.arange(X.shape[0])
    X_train, X_test, y_local_train, y_local_test, indices_train, indices_test = train_test_split(
        X, y_local, indices, test_size=0.20, random_state=42
    )

    y_visitante_train = y_visitante.iloc[indices_train]
    y_visitante_test = y_visitante.iloc[indices_test]
    y_1x2_test = y_1x2_real.iloc[indices_test]

    # 3. Escalado de Características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Instanciación y Ajuste de los Modelos de Poisson
    print("[MODELO] Entrenando Regresores de Poisson...")
    modelo_local = PoissonRegressor(alpha=1e-4, max_iter=300)
    modelo_visitante = PoissonRegressor(alpha=1e-4, max_iter=300)

    modelo_local.fit(X_train_scaled, y_local_train)
    modelo_visitante.fit(X_train_scaled, y_visitante_train)

    # ============================================================
    # MÓDULO 4: EVALUACIÓN DE MÉTRICAS DE NEGOCIO (MUNDIAL)
    # ============================================================

    pred_lambda_local = modelo_local.predict(X_test_scaled)
    pred_lambda_visitante = modelo_visitante.predict(X_test_scaled)

    mae_local = mean_absolute_error(y_local_test, pred_lambda_local)
    mae_local_redondeado = mean_absolute_error(
        y_local_test, np.round(pred_lambda_local))
    mae_visitante = mean_absolute_error(
        y_visitante_test, pred_lambda_visitante)

    print("\n" + "="*50)
    print("      MÉTRICAS DE RENDIMIENTO DE GOLES (REGRESIÓN)")
    print("="*50)
    print(f"🔹 MAE Goles Local (Continuo):    {mae_local:.3f} goles")
    print(f"🔹 MAE Goles Local (Redondeado):  {mae_local_redondeado:.3f} goles")
    print(f"🔹 MAE Goles Visitante:           {mae_visitante:.3f} goles")

    # CORRECCIÓN AQUÍ: Cambiamos los nombres de las columnas para coincidir con lo que busca 'definir_resultado_1x2'
    df_preds_test = pd.DataFrame({
        'goles_local': np.round(pred_lambda_local),
        'goles_visitante': np.round(pred_lambda_visitante)
    })
    y_1x2_pred = df_preds_test.apply(definir_resultado_1x2, axis=1)

    accuracy_1x2 = accuracy_score(y_1x2_test, y_1x2_pred)

    print("\n" + "="*50)
    print("     MÉTRICAS DE RENDIMIENTO TIPO DE APUESTA (1X2)")
    print("="*50)
    print(f"⚽ Precisión Global (Accuracy 1X2): {accuracy_1x2 * 100:.2f}%")
    print("\n[REPORTE DE CLASIFICACIÓN DETALLADO]:")
    print(classification_report(y_1x2_test, y_1x2_pred, target_names=[
          'Gana Local (1)', 'Empate (X)', 'Gana Visitante (2)'], zero_division=0))

    return modelo_local, modelo_visitante, scaler, features

# ============================================================
# MÓDULO 5: EXPORTACIÓN Y PERSISTENCIA
# ============================================================


def guardar_artefactos(modelo_l, modelo_v, scaler, features, ruta_salida):
    """
    Empaqueta los modelos, el transformador de datos y los metadatos en un único
    archivo binario para asegurar que el script de predicción final sea idéntico.
    """
    print(f"[EXPORT] Guardando modelos y escalador en '{ruta_salida}'...")
    artefactos = {
        'modelo_local': modelo_l,
        'modelo_visitante': modelo_v,
        'scaler': scaler,
        'features_names': features,
        'metadata': {
            'algoritmo': 'PoissonRegressor (Dual Ensemble)',
            'optimizador': 'L-BFGS-B'
        }
    }
    joblib.dump(artefactos, ruta_salida)
    print("[OK] Serialización completada con éxito.")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("     PIPELINE DE ENTRENAMIENTO - MUNDIAL 2026")
    print("="*50)

    archivo_datos = "dataset_mundial_limpio.csv"
    df_partidos = cargar_y_construir_partidos(archivo_datos)

    mod_local, mod_visitante, sc, feat = entrenar_sistema_predictivo(
        df_partidos)

    guardar_artefactos(mod_local, mod_visitante, sc,
                       feat, "modelo_prediccion_2026.pkl")
