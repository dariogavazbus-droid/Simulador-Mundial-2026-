import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from collections import Counter

# CONFIGURACIÓN DE PÁGINA (Esto hace que se vea bien en el celu)
st.set_page_config(page_title="Simulador Mundial 2026",
                   page_icon="🏆", layout="centered")

# ============================================================
# MÓDULO 1: BASE DE DATOS Y CONFIGURACIÓN
# ============================================================

MAPA_SEDES = {
    "0": {"nombre": "Nivel del mar / Estándar", "altitud": 0},
    "1": {"nombre": "Estadio Azteca (CDMX)", "altitud": 2240},
    "2": {"nombre": "Estadio Akron (Guadalajara)", "altitud": 1566}
}

DB_EQUIPOS = {
    "Alemania": {"ranking_fifa": 16, "goles_favor": 1.9, "goles_contra": 1.2, "estilo": 1},
    "Arabia Saudita": {"ranking_fifa": 53, "goles_favor": 1.1, "goles_contra": 1.4, "estilo": 2},
    "Argelia": {"ranking_fifa": 43, "goles_favor": 1.3, "goles_contra": 1.2, "estilo": 2},
    "Argentina": {"ranking_fifa": 1, "goles_favor": 2.4, "goles_contra": 0.6, "estilo": 1},
    "Australia": {"ranking_fifa": 24, "goles_favor": 1.3, "goles_contra": 1.1, "estilo": 2},
    "Austria": {"ranking_fifa": 25, "goles_favor": 1.4, "goles_contra": 1.2, "estilo": 1},
    "Belgica": {"ranking_fifa": 8, "goles_favor": 1.8, "goles_contra": 1.0, "estilo": 1},
    "Bosnia": {"ranking_fifa": 75, "goles_favor": 1.1, "goles_contra": 1.4, "estilo": 2},
    "Brasil": {"ranking_fifa": 5, "goles_favor": 2.0, "goles_contra": 0.9, "estilo": 1},
    "Cabo Verde": {"ranking_fifa": 65, "goles_favor": 1.1, "goles_contra": 1.3, "estilo": 2},
    "Canada": {"ranking_fifa": 40, "goles_favor": 1.3, "goles_contra": 1.3, "estilo": 2},
    "Chequia": {"ranking_fifa": 35, "goles_favor": 1.3, "goles_contra": 1.2, "estilo": 2},
    "Colombia": {"ranking_fifa": 12, "goles_favor": 1.7, "goles_contra": 1.0, "estilo": 1},
    "Congo": {"ranking_fifa": 66, "goles_favor": 1.1, "goles_contra": 1.3, "estilo": 2},
    "Corea Del Sur": {"ranking_fifa": 22, "goles_favor": 1.4, "goles_contra": 1.1, "estilo": 2},
    "Costa De Marfil": {"ranking_fifa": 39, "goles_favor": 1.3, "goles_contra": 1.1, "estilo": 2},
    "Croacia": {"ranking_fifa": 13, "goles_favor": 1.5, "goles_contra": 1.1, "estilo": 1},
    "Curazao": {"ranking_fifa": 88, "goles_favor": 1.0, "goles_contra": 1.5, "estilo": 2},
    "Ecuador": {"ranking_fifa": 30, "goles_favor": 1.3, "goles_contra": 1.1, "estilo": 2},
    "Egipto": {"ranking_fifa": 36, "goles_favor": 1.3, "goles_contra": 1.1, "estilo": 2},
    "Escocia": {"ranking_fifa": 34, "goles_favor": 1.2, "goles_contra": 1.3, "estilo": 2},
    "Espana": {"ranking_fifa": 3, "goles_favor": 2.1, "goles_contra": 0.8, "estilo": 1},
    "Estados Unidos": {"ranking_fifa": 14, "goles_favor": 1.6, "goles_contra": 1.1, "estilo": 1},
    "Francia": {"ranking_fifa": 2, "goles_favor": 2.2, "goles_contra": 0.8, "estilo": 1},
    "Ghana": {"ranking_fifa": 60, "goles_favor": 1.2, "goles_contra": 1.4, "estilo": 2},
    "Jordania": {"ranking_fifa": 68, "goles_favor": 1.1, "goles_contra": 1.3, "estilo": 2},
    "Haiti": {"ranking_fifa": 88, "goles_favor": 1.0, "goles_contra": 1.5, "estilo": 2},
    "Inglaterra": {"ranking_fifa": 4, "goles_favor": 2.1, "goles_contra": 0.7, "estilo": 1},
    "Iran": {"ranking_fifa": 20, "goles_favor": 1.3, "goles_contra": 1.0, "estilo": 2},
    "Irak": {"ranking_fifa": 55, "goles_favor": 1.1, "goles_contra": 1.3, "estilo": 2},
    "Japon": {"ranking_fifa": 17, "goles_favor": 1.6, "goles_contra": 1.0, "estilo": 1},
    "Marruecos": {"ranking_fifa": 13, "goles_favor": 1.5, "goles_contra": 0.9, "estilo": 2},
    "Mexico": {"ranking_fifa": 15, "goles_favor": 1.4, "goles_contra": 1.2, "estilo": 1},
    "Noruega": {"ranking_fifa": 33, "goles_favor": 1.4, "goles_contra": 1.2, "estilo": 2},
    "Nueva Zelanda": {"ranking_fifa": 85, "goles_favor": 1.0, "goles_contra": 1.5, "estilo": 2},
    "Paises Bajos": {"ranking_fifa": 7, "goles_favor": 1.9, "goles_contra": 1.0, "estilo": 1},
    "Panama": {"ranking_fifa": 45, "goles_favor": 1.2, "goles_contra": 1.2, "estilo": 2},
    "Paraguay": {"ranking_fifa": 56, "goles_favor": 1.0, "goles_contra": 1.2, "estilo": 2},
    "Portugal": {"ranking_fifa": 6, "goles_favor": 2.0, "goles_contra": 0.9, "estilo": 1},
    "Qatar": {"ranking_fifa": 46, "goles_favor": 1.2, "goles_contra": 1.3, "estilo": 2},
    "Senegal": {"ranking_fifa": 18, "goles_favor": 1.4, "goles_contra": 1.0, "estilo": 2},
    "Sudafrica": {"ranking_fifa": 59, "goles_favor": 1.1, "goles_contra": 1.3, "estilo": 2},
    "Suecia": {"ranking_fifa": 28, "goles_favor": 1.4, "goles_contra": 1.2, "estilo": 1},
    "Suiza": {"ranking_fifa": 19, "goles_favor": 1.4, "goles_contra": 1.2, "estilo": 1},
    "Tunez": {"ranking_fifa": 41, "goles_favor": 1.0, "goles_contra": 1.2, "estilo": 2},
    "Turquía": {"ranking_fifa": 42, "goles_favor": 1.4, "goles_contra": 1.2, "estilo": 1},
    "Uruguay": {"ranking_fifa": 11, "goles_favor": 1.8, "goles_contra": 1.0, "estilo": 1},
    "Uzbekistan": {"ranking_fifa": 60, "goles_favor": 1.2, "goles_contra": 1.3, "estilo": 2}
}

# ============================================================
# MÓDULO 2: CARGA DE ARTEFACTOS (OPTIMIZADO)
# ============================================================

@st.cache_resource
def cargar_sistema_predictivo():
    # Obtiene la ruta de la carpeta donde se encuentra app.py
    carpeta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(carpeta_actual, 'modelo_prediccion_2026.pkl')
    
    if not os.path.exists(ruta_modelo):
        st.error(f"No se encontró el archivo en: {ruta_modelo}")
        st.stop()
    
    return joblib.load(ruta_modelo)

def construir_vector_features(eq_l, eq_v):
    stats_l = DB_EQUIPOS[eq_l]
    stats_v = DB_EQUIPOS[eq_v]
    data_dict = {
        'ranking_fifa_local': stats_l['ranking_fifa'],
        'ranking_fifa_visitante': stats_v['ranking_fifa'],
        'promedio_goles_favor_u10_local': stats_l['goles_favor'],
        'promedio_goles_favor_u10_visitante': stats_v['goles_favor'],
        'promedio_goles_contra_u10_local': stats_l['goles_contra'],
        'promedio_goles_contra_u10_visitante': stats_v['goles_contra']
    }
    cols = ['ranking_fifa_local', 'ranking_fifa_visitante', 'promedio_goles_favor_u10_local',
            'promedio_goles_favor_u10_visitante', 'promedio_goles_contra_u10_local', 'promedio_goles_contra_u10_visitante']
    return pd.DataFrame([data_dict])[cols]

# ============================================================
# MÓDULO 3: MOTOR DE SIMULACIÓN
# ============================================================


def simular_partido_monte_carlo(eq_1, eq_2, artefactos, urgencia_1, urgencia_2, altitud, n_simulaciones=10000):
    estilo_1 = DB_EQUIPOS[eq_1]["estilo"]
    estilo_2 = DB_EQUIPOS[eq_2]["estilo"]

    df_A = construir_vector_features(eq_1, eq_2)
    df_B = construir_vector_features(eq_2, eq_1)

    X_A = artefactos['scaler'].transform(df_A)
    X_B = artefactos['scaler'].transform(df_B)

    l1_A = artefactos['modelo_local'].predict(X_A)[0]
    l2_V = artefactos['modelo_visitante'].predict(X_A)[0]
    l2_L = artefactos['modelo_local'].predict(X_B)[0]
    l1_V = artefactos['modelo_visitante'].predict(X_B)[0]

    lambda_final_1 = (l1_A + l1_V) / 2
    lambda_final_2 = (l2_L + l2_V) / 2

    if altitud > 0:
        factor_alt = 1 + (altitud / 15000)
        lambda_final_1 *= factor_alt
        lambda_final_2 *= factor_alt

    lambda_final_1 *= 1.05 if estilo_1 == 1 else 0.90
    lambda_final_2 *= 1.05 if estilo_2 == 1 else 0.90

    if urgencia_1 == 3:
        lambda_final_1 *= 1.30
        lambda_final_2 *= 1.15
    elif urgencia_1 == 1:
        lambda_final_1 *= 0.80
        lambda_final_2 *= 0.85
    if urgencia_2 == 3:
        lambda_final_2 *= 1.30
        lambda_final_1 *= 1.15
    elif urgencia_2 == 1:
        lambda_final_2 *= 0.80
        lambda_final_1 *= 0.85

    np.random.seed(42)
    g1 = np.random.poisson(lambda_final_1, n_simulaciones)
    g2 = np.random.poisson(lambda_final_2, n_simulaciones)

    p1 = (np.sum(g1 > g2)/n_simulaciones)*100
    pE = (np.sum(g1 == g2)/n_simulaciones)*100
    p2 = (np.sum(g1 < g2)/n_simulaciones)*100

    marcador = Counter(list(zip(g1, g2))).most_common(1)[0][0]

    return {'p1': p1, 'pE': pE, 'p2': p2, 'marcador': marcador}

# ============================================================
# MÓDULO 4: INTERFAZ WEB (STREAMLIT)
# ============================================================


st.title("🏆 Simulador Mundial 2026")
st.write("Predicción probabilística basada en modelos de Poisson y Monte Carlo.")

# Cargar cerebro
artefactos = cargar_sistema_predictivo()

# Sidebar para Sede
st.sidebar.header("🏟️ Configuración del Estadio")
sede_nombre = st.sidebar.selectbox(
    "Seleccioná la sede", [v['nombre'] for v in MAPA_SEDES.values()])
info_sede = next(v for v in MAPA_SEDES.values() if v['nombre'] == sede_nombre)

# Cuerpo Principal
equipos_lista = sorted(list(DB_EQUIPOS.keys()))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Equipo Local")
    eq1 = st.selectbox("Seleccioná Equipo 1", equipos_lista,
                       index=equipos_lista.index("Argentina"))
    urg1 = st.select_slider(f"Urgencia de {eq1}", options=[1, 2, 3], value=2,
                            format_func=lambda x: {1: "Conservador", 2: "Normal", 3: "¡Obligado!"}[x])

with col2:
    st.subheader("Equipo Visitante")
    eq2 = st.selectbox("Seleccioná Equipo 2", equipos_lista,
                       index=equipos_lista.index("Brasil"))
    urg2 = st.select_slider(f"Urgencia de {eq2}", options=[1, 2, 3], value=2,
                            format_func=lambda x: {1: "Conservador", 2: "Normal", 3: "¡Obligado!"}[x])

st.divider()

if st.button("🚀 SIMULAR PARTIDO", use_container_width=True):
    if eq1 == eq2:
        st.warning("⚠️ Seleccioná dos equipos diferentes.")
    else:
        with st.spinner('Procesando 10,000 iteraciones...'):
            res = simular_partido_monte_carlo(
                eq1, eq2, artefactos, urg1, urg2, info_sede['altitud'])

            # Mostrar Resultados Pro
            st.success("¡Simulación Exitosa!")

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Prob. {eq1}", f"{res['p1']:.1f}%")
            c2.metric("Empate", f"{res['pE']:.1f}%")
            c3.metric(f"Prob. {eq2}", f"{res['p2']:.2f}%")

            st.info(
                f"🎯 **Marcador más probable:** {eq1} {res['marcador'][0]} - {res['marcador'][1]} {eq2}")

st.sidebar.info(
    f"Sede: {info_sede['nombre']}\nAltitud: {info_sede['altitud']}m")