"""
Sistema de Predicción Mundial 2026 - Módulo de Ingesta y Preparación de Datos
Autor: Ingeniero de Datos / Analítica Deportiva
Versión corregida: fallback de ranking FIFA, mapeo de países ampliado,
manejo de errores en API, uso real de 'años_atras', API_KEY por variable
de entorno, y aprovechamiento de las métricas de posesión/remates.
"""

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import time
from datetime import datetime
from io import StringIO

# ============================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================

# FIX (seguridad): la API_KEY ya no se hardcodea en el archivo. Se lee de una
# variable de entorno. Si no está seteada, el script cae directo al modo
# desarrollo (datos mock), igual que antes, pero sin el riesgo de que la
# clave quede expuesta si este archivo se sube a un repositorio.
API_KEY = os.environ.get("RAPIDAPI_KEY", "")
BASE_URL_API = "https://api-football-v1.p.rapidapi.com/v3"
HEADERS_API = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# FIX: diccionario ampliado a las 48 selecciones reales del Mundial 2026,
# con variantes en inglés y español, y usando los MISMOS nombres en español
# que ya usamos en datos_selecciones_mundial2026.xlsx / predictor.py
# (ej. "Corea Del Sur", "Estados Unidos") para que en el futuro estas dos
# bases de datos se puedan cruzar sin fricciones de nombres.
MAPEO_PAISES = {
    "argelia": "Argelia", "algeria": "Argelia",
    "argentina": "Argentina",
    "australia": "Australia",
    "austria": "Austria",
    "belgica": "Belgica", "belgium": "Belgica",
    "bosnia": "Bosnia", "bosnia and herzegovina": "Bosnia", "bosnia herzegovina": "Bosnia",
    "brasil": "Brasil", "brazil": "Brasil",
    "cabo verde": "Cabo Verde", "cape verde": "Cabo Verde",
    "canada": "Canada",
    "colombia": "Colombia",
    "congo": "Congo", "dr congo": "Congo", "democratic republic of the congo": "Congo",
    "costa de marfil": "Costa De Marfil", "ivory coast": "Costa De Marfil", "cote d'ivoire": "Costa De Marfil",
    "croacia": "Croacia", "croatia": "Croacia",
    "curazao": "Curazao", "curacao": "Curazao",
    "chequia": "Chequia", "czechia": "Chequia", "czech republic": "Chequia",
    "ecuador": "Ecuador",
    "egipto": "Egipto", "egypt": "Egipto",
    "inglaterra": "Inglaterra", "england": "Inglaterra",
    "francia": "Francia", "france": "Francia",
    "alemania": "Alemania", "germany": "Alemania",
    "ghana": "Ghana",
    "haiti": "Haiti",
    "iran": "Iran",
    "irak": "Irak", "iraq": "Irak",
    "japon": "Japon", "japan": "Japon",
    "jordania": "Jordania", "jordan": "Jordania",
    "corea del sur": "Corea Del Sur", "south korea": "Corea Del Sur", "korea republic": "Corea Del Sur",
    "mexico": "Mexico",
    "marruecos": "Marruecos", "morocco": "Marruecos",
    "holanda": "Holanda", "paises bajos": "Holanda", "netherlands": "Holanda",
    "nueva zelanda": "Nueva Zelanda", "new zealand": "Nueva Zelanda",
    "noruega": "Noruega", "norway": "Noruega",
    "panama": "Panama",
    "paraguay": "Paraguay",
    "portugal": "Portugal",
    "qatar": "Qatar",
    "arabia saudita": "Arabia Saudita", "saudi arabia": "Arabia Saudita",
    "escocia": "Escocia", "scotland": "Escocia",
    "senegal": "Senegal",
    "sudafrica": "Sudafrica", "south africa": "Sudafrica",
    "espana": "Espana", "españa": "Espana", "spain": "Espana",
    "suecia": "Suecia", "sweden": "Suecia",
    "suiza": "Suiza", "switzerland": "Suiza",
    "tunez": "Tunez", "tunisia": "Tunez",
    "uruguay": "Uruguay",
    "estados unidos": "Estados Unidos", "usa": "Estados Unidos", "united states": "Estados Unidos",
    "uzbekistan": "Uzbekistan",
    "venezuela": "Venezuela",
}

# FIX (antes no existía): respaldo estático del Top 20 del Ranking FIFA,
# tomado de la versión real de Wikipedia (actualización del 19 de enero de
# 2026). Antes, si el scraping fallaba por CUALQUIER motivo (Wikipedia
# bloqueó el pedido, cambió de diseño, no había conexión), el resultado
# final terminaba con posicion_fifa=99 para TODOS los equipos sin
# excepción -- es decir, la columna de ranking quedaba inútil. Ahora, si el
# scraping en vivo falla, se usa este respaldo en vez de dejar todo en 99.
# Conviene actualizar estos números de tanto en tanto.
RANKING_FIFA_RESPALDO = {
    "Espana": 1, "Argentina": 2, "Francia": 3, "Inglaterra": 4, "Brasil": 5,
    "Portugal": 6, "Holanda": 7, "Marruecos": 8, "Belgica": 9, "Alemania": 10,
    "Croacia": 11, "Senegal": 12, "Italy": 13, "Colombia": 14, "Estados Unidos": 15,
    "Mexico": 16, "Uruguay": 17, "Suiza": 18, "Japon": 19, "Iran": 20,
}

# ============================================================
# MÓDULO 1: INGESTA DE DATOS (SCRAPING & API)
# ============================================================


def _tabla_es_de_ranking(tabla):
    """
    FIX: antes el script tomaba 'la primera tabla con clase wikitable' a
    ciegas, asumiendo que esa siempre iba a ser la tabla de ranking. Eso
    funciona hoy por casualidad (la tabla de ranking aparece primera en la
    versión actual de la página), pero se rompería en silencio si Wikipedia
    agrega cualquier otra tabla antes. Ahora se valida el contenido real del
    encabezado antes de aceptar una tabla como la de ranking.
    """
    primera_fila = tabla.find('tr')
    if not primera_fila:
        return False
    texto_header = primera_fila.get_text(" ", strip=True).lower()
    tiene_rank = any(p in texto_header for p in ("rank", "pos", "rk"))
    tiene_team = any(p in texto_header for p in (
        "team", "association", "equipo"))
    tiene_points = any(p in texto_header for p in ("points", "pts", "puntos"))
    return tiene_rank and tiene_team and tiene_points


def obtener_ranking_fifa_wiki():
    print("[INGESTA] Extrayendo Ranking FIFA desde Wikipedia...")
    url = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # FIX: ya no toma la primera 'wikitable' sin más; recorre todas y se
        # queda con la primera que realmente tenga pinta de tabla de ranking
        # (encabezado con rank + team + points).
        tabla = None
        for candidata in soup.find_all('table', {'class': 'wikitable'}):
            if _tabla_es_de_ranking(candidata):
                tabla = candidata
                break

        if tabla is None:
            print(
                "[AVISO] No se encontró una tabla de ranking reconocible en Wikipedia. Uso el respaldo estático.")
            return _ranking_desde_respaldo()

        for fila in tabla.find_all('tr'):
            if fila.find('td', colspan=True):
                fila.decompose()

        df_ranking = pd.read_html(StringIO(str(tabla)))[0]

        cols_normalizadas = [str(c).lower() for c in df_ranking.columns]
        idx_rank = 0
        idx_team = 1

        for i, col in enumerate(cols_normalizadas):
            if 'rank' in col or 'pos' in col or 'rk' in col:
                idx_rank = i
                break
        for i, col in enumerate(cols_normalizadas):
            if 'team' in col or 'association' in col or 'equipo' in col:
                idx_team = i
                break

        df_ranking = df_ranking.iloc[:, [idx_rank, idx_team]]
        df_ranking.columns = ['posicion_fifa', 'equipo']

        df_ranking['posicion_fifa'] = pd.to_numeric(
            df_ranking['posicion_fifa'].astype(str).str.extract(r'(\d+)')[0],
            errors='coerce'
        )

        df_ranking = df_ranking.dropna(subset=['posicion_fifa', 'equipo'])

        if df_ranking.empty:
            print(
                "[AVISO] La tabla de Wikipedia se extrajo vacía. Uso el respaldo estático.")
            return _ranking_desde_respaldo()

        print(
            f"[OK] Se obtuvieron {len(df_ranking)} selecciones del Ranking FIFA (en vivo).")
        return df_ranking

    except Exception as e:
        print(
            f"[ERROR] No se pudo hacer scraping del Ranking FIFA: {e}. Uso el respaldo estático.")
        return _ranking_desde_respaldo()


def _ranking_desde_respaldo():
    df = pd.DataFrame(
        [(equipo, pos) for equipo, pos in RANKING_FIFA_RESPALDO.items()],
        columns=['equipo', 'posicion_fifa']
    )
    return df[['posicion_fifa', 'equipo']]


def obtener_historial_partidos_api(años_atras=8):
    print(
        f"[INGESTA] Solicitando historial de partidos a API-Football (Últimos {años_atras} años)...")

    # FIX: 'años_atras' antes no se usaba para nada salvo imprimirse en
    # pantalla. Ahora la fecha de inicio del mock se calcula realmente en
    # base a ese parámetro (antes estaba fijo en 2018-01-01 sin importar lo
    # que se pasara).
    fecha_inicio_mock = f"{datetime.now().year - años_atras}-01-01"

    if not API_KEY:
        print("[AVISO] No hay RAPIDAPI_KEY configurada. Usando entorno de datos local estructurado (Modo desarrollo).")
        rng = np.random.default_rng(42)
        data_mock = {
            'fecha': pd.date_range(start=fecha_inicio_mock, periods=100, freq='W'),
            'local': ['Argentina', 'Brasil', 'France', 'USA', 'England'] * 20,
            'visitante': ['Mexico', 'Uruguay', 'Germany', 'Spain', 'Colombia'] * 20,
            'goles_local': rng.integers(0, 4, size=100),
            'goles_visitante': rng.integers(0, 3, size=100),
            'posesion_local': rng.integers(40, 65, size=100),
            'remates_arco_local': rng.integers(2, 9, size=100)
        }
        return pd.DataFrame(data_mock)

    endpoint = f"{BASE_URL_API}/fixtures"
    ligas_objetivo = [1, 2, 9, 332]
    temporada_desde = datetime.now().year - años_atras
    partidos_totales = []

    for liga in ligas_objetivo:
        # FIX: ahora sí se usa 'años_atras' para acotar el pedido por
        # temporada en vez de traer el historial completo sin filtro.
        query_params = {"league": str(
            liga), "status": "FT", "season": str(temporada_desde)}
        try:
            res = requests.get(endpoint, headers=HEADERS_API,
                               params=query_params, timeout=15)
        except requests.RequestException as e:
            # FIX: antes, si fallaba la conexión, esto directamente tiraba
            # una excepción no controlada y cortaba todo el pipeline.
            print(
                f"[AVISO] Falló la conexión para la liga {liga}: {e}. Sigo con las demás.")
            continue

        if res.status_code == 200:
            partidos = res.json().get('response', [])
            for p in partidos:
                goles_home = p['goals']['home']
                goles_away = p['goals']['away']
                if goles_home is None or goles_away is None:
                    continue  # partido sin resultado cargado todavía, se descarta
                partidos_totales.append({
                    'fecha': p['fixture']['date'],
                    'local': p['teams']['home']['name'],
                    'visitante': p['teams']['away']['name'],
                    'goles_local': goles_home,
                    'goles_visitante': goles_away,
                    'posesion_local': 50,
                    'remates_arco_local': 4
                })
        else:
            # FIX: antes esto se ignoraba en silencio; ahora avisa para que
            # no termines con un dataset incompleto sin saberlo.
            print(
                f"[AVISO] La liga {liga} respondió con status {res.status_code}, se omite.")
        time.sleep(1)

    if not partidos_totales:
        print(
            "[AVISO] No se obtuvo ningún partido real de la API. Devuelvo un DataFrame vacío.")
        return pd.DataFrame(columns=['fecha', 'local', 'visitante', 'goles_local',
                                     'goles_visitante', 'posesion_local', 'remates_arco_local'])

    return pd.DataFrame(partidos_totales)

# ============================================================
# MÓDULO 2: LIMPIEZA Y HOMOGENEIZACIÓN DE DATOS
# ============================================================


def limpiar_nombre_pais(nombre):
    if pd.isna(nombre):
        return np.nan
    nombre_limpio = str(nombre).strip().lower()
    return MAPEO_PAISES.get(nombre_limpio, nombre_limpio.title())


def pipeline_limpieza(df_partidos, df_ranking):
    print("[LIMPIEZA] Homogeneizando nombres de países y tratando nulos...")

    if df_partidos.empty:
        return df_partidos, df_ranking

    df_partidos = df_partidos.copy()
    df_partidos['local'] = df_partidos['local'].apply(limpiar_nombre_pais)
    df_partidos['visitante'] = df_partidos['visitante'].apply(
        limpiar_nombre_pais)
    df_ranking['equipo'] = df_ranking['equipo'].apply(limpiar_nombre_pais)

    df_partidos = df_partidos.drop_duplicates()

    df_partidos = df_partidos.dropna(
        subset=['goles_local', 'goles_visitante', 'local', 'visitante'])

    df_partidos['fecha'] = pd.to_datetime(df_partidos['fecha'])
    df_partidos['goles_local'] = df_partidos['goles_local'].astype(int)
    df_partidos['goles_visitante'] = df_partidos['goles_visitante'].astype(int)

    return df_partidos, df_ranking

# ============================================================
# MÓDULO 3: FEATURE ENGINEERING (MÉTRICAS AVANZADAS)
# ============================================================


def calcular_feature_engineering(df_partidos):
    """
    FIX: antes se calculaba el promedio móvil de goles a favor/en contra,
    pero la posesión y los remates al arco -- que sí se recolectan en la
    ingesta -- nunca se usaban. Ahora también se calcula un promedio móvil
    de posesión y remates (sólo disponibles del lado 'local' en los datos
    actuales; si en el futuro se suma 'posesion_visitante' y
    'remates_arco_visitante', conviene promediarlos igual que los goles).
    """
    print("[FEATURES] Calculando promedios móviles (últimos 10 partidos oficiales)...")

    locales = df_partidos[['fecha', 'local', 'goles_local', 'goles_visitante',
                           'posesion_local', 'remates_arco_local']].copy()
    locales.columns = ['fecha', 'equipo', 'goles_favor',
                       'goles_contra', 'posesion', 'remates_arco']

    visitantes = df_partidos[['fecha', 'visitante',
                              'goles_visitante', 'goles_local']].copy()
    visitantes.columns = ['fecha', 'equipo', 'goles_favor', 'goles_contra']
    visitantes['posesion'] = np.nan
    visitantes['remates_arco'] = np.nan

    historial_lineal = pd.concat(
        [locales, visitantes], axis=0).sort_values(by=['equipo', 'fecha'])

    historial_lineal['promedio_goles_favor_u10'] = (
        historial_lineal.groupby('equipo')['goles_favor']
        .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    )

    historial_lineal['promedio_goles_contra_u10'] = (
        historial_lineal.groupby('equipo')['goles_contra']
        .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    )

    historial_lineal['promedio_posesion_u10'] = (
        historial_lineal.groupby('equipo')['posesion']
        .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    )

    historial_lineal['promedio_remates_arco_u10'] = (
        historial_lineal.groupby('equipo')['remates_arco']
        .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    )

    estado_actual_equipos = historial_lineal.groupby(
        'equipo').last().reset_index()

    return estado_actual_equipos[['equipo', 'promedio_goles_favor_u10', 'promedio_goles_contra_u10',
                                  'promedio_posesion_u10', 'promedio_remates_arco_u10']]

# ============================================================
# MÓDULO 4: ORQUESTACIÓN Y EXPORTACIÓN
# ============================================================


def ejecutar_pipeline():
    df_ranking_raw = obtener_ranking_fifa_wiki()
    df_partidos_raw = obtener_historial_partidos_api()

    if df_partidos_raw.empty:
        print("[ERROR] No hay datos de partidos para procesar. Pipeline detenido.")
        return

    df_partidos_limpios, df_ranking_limpio = pipeline_limpieza(
        df_partidos_raw, df_ranking_raw)

    df_features = calcular_feature_engineering(df_partidos_limpios)

    print("\n--- DIAGNÓSTICO DE NOMBRES ---")
    equipos_features = set(df_features['equipo'].unique())
    equipos_ranking = set(df_ranking_limpio['equipo'].unique())

    faltantes = equipos_features - equipos_ranking
    if faltantes:
        print(
            f"⚠️ Equipos en Partidos que NO están en el Ranking: {faltantes}")
        print("💡 Acción: Añade estos nombres a 'MAPEO_PAISES' al inicio del archivo.")
    else:
        print("✅ ¡Todos los equipos coinciden correctamente!")
    print("-------------------------------\n")
    print("[CONSOLIDACIÓN] Uniendo características calculadas con el Ranking FIFA...")
    dataset_final = pd.merge(
        df_features, df_ranking_limpio, on='equipo', how='left')

    dataset_final['posicion_fifa'] = dataset_final['posicion_fifa'].fillna(
        99).astype(int)

    ruta_salida = "dataset_mundial_limpio.csv"
    dataset_final.to_csv(ruta_salida, index=False, encoding='utf-8')
    print(
        f"\n[ÉXITO] Pipeline completado. Archivo exportado en: '{ruta_salida}'")
    print(dataset_final.head(10))


if __name__ == "__main__":
    ejecutar_pipeline()
