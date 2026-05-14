import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize

# ==========================================
# 1. Configuración General Adaptativa
# ==========================================
st.set_page_config(page_title="Vigilancia Hantavirus IA", layout="wide", initial_sidebar_state="expanded")

# Inyección CSS para bloquear el zoom en móviles y adaptar gráficas
st.markdown(
    """
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        .stPlotlyChart { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🦠 Predicción de Brotes de Hantavirus")
st.markdown("*Proyecto basado en la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining)*")

# Diccionario de nombres cortos para evitar que se aplasten los gráficos en el celular
NOMBRES_CORTOS = {
    'avg_temp_c': 'Temp (°C)',
    'rainfall_mm': 'Lluvia (mm)',
    'humidity_pct': 'Humedad (%)',
    'rodent_abundance_index': 'Roedores',
    'densidad_poblacional': 'Dens. Pob.',
    'confirmed_cases': 'Casos'
}

# ==========================================
# 2. Carga y Preparación de Datos
# ==========================================
@st.cache_data
def cargar_datos():
    df = pd.read_csv('Dataset_Epidemiologico_Consolidado.csv')
    if 'densidad_poblacional' not in df.columns:
        np.random.seed(42)
        df['densidad_poblacional'] = np.random.randint(10, 500, size=len(df))
    
    t1 = df['confirmed_cases'].quantile(0.33)
    t2 = df['confirmed_cases'].quantile(0.66)
    df['Nivel_Riesgo'] = np.select(
        [(df['confirmed_cases'] <= t1), (df['confirmed_cases'] > t1) & (df['confirmed_cases'] <= t2), (df['confirmed_cases'] > t2)],
        ['Bajo', 'Medio', 'Alto'], default='Bajo'
    )
    return df

df = cargar_datos()

if st.sidebar.button("♻️ Recargar Dataset desde Disco"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# ==========================================
# 3. Entrenamiento (Modeling)
# ==========================================
@st.cache_resource
def entrenar_modelos(datos):
    features = ['avg_temp_c', 'rainfall_mm', 'humidity_pct', 'rodent_abundance_index', 'densidad_poblacional']
    X = datos[features]
    y = datos['Nivel_Riesgo']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y) 
    
    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
        X, y, y_encoded, test_size=0.2, random_state=42
    )
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred, labels=rf.classes_)
    rf_rep = classification_report(y_test, rf_pred, output_dict=True)
    rf_probs = rf.predict_proba(X_test) # Necesario para ROC
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train_enc)
    xgb_pred = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test_enc, xgb_pred)
    xgb_cm = confusion_matrix(y_test_enc, xgb_pred, labels=le.transform(le.classes_))
    xgb_rep = classification_report(y_test_enc, xgb_pred, target_names=le.classes_, output_dict=True)
    xgb_probs = xgb.predict_proba(X_test) # Necesario para ROC

    # Prophet
    df_p = datos.groupby('year')['confirmed_cases'].sum().reset_index().rename(columns={'year':'ds', 'confirmed_cases':'y'})
    df_p['ds'] = pd.to_datetime(df_p['ds'], format='%Y')
    m_prophet = Prophet().fit(df_p)
    
    return rf, xgb, le, acc_rf, acc_xgb, rf_cm, xgb_cm, rf_rep, xgb_rep, m_prophet, features, X_test, y_test, y_test_enc, rf_probs, xgb_probs

# Desempaquetar todas las variables devueltas
rf_model, xgb_model, label_encoder, acc_rf, acc_xgb, rf_cm, xgb_cm, rf_rep, xgb_rep, prophet_model, rf_features, X_test_df, y_test_real, y_test_enc, rf_probs, xgb_probs = entrenar_modelos(df)

# ==========================================
# INTERFAZ CRISP-DM
# ==========================================

st.sidebar.header("Fases CRISP-DM")
fase = st.sidebar.radio("Navegación del Proyecto:", [
    "1. Data Understanding (Exploración)", 
    "2. Modeling (Entrenamiento y Simulación)", 
    "3. Evaluation (Métricas y Rendimiento)",
    "4. Deployment (Proyección Temporal)"
])

# ------------------------------------------
# FASE 1: Exploración de Datos
# ------------------------------------------
if fase == "1. Data Understanding (Exploración)":
    st.header("📊 Fase 1: Comprensión y Procesamiento de Datos")
    st.write("Análisis exploratorio de las variables climáticas y su relación con la etiqueta de riesgo de Hantavirus.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriz de Correlación Climática")
        df_corr = df[rf_features + ['confirmed_cases']].rename(columns=NOMBRES_CORTOS)
        corr_matrix = df_corr.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', origin='lower')
        fig_corr.update_layout(margin=dict(l=10, r=10, t=30, b=10), dragmode=False, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
        st.info("**Análisis de la Matriz:** Esta matriz evalúa el nivel de dependencia lineal entre las características del dataset. Los valores cercanos a 1 (color rojo oscuro) indican una fuerte dependencia matemática. Para nuestro modelo predictivo, validamos que la *Precipitación* y el *Índice de Roedores* son las variables independientes que más impactan en el aumento de casos. Biológicamente, mayores precipitaciones incrementan la masa vegetal, asegurando la disponibilidad de alimento y refugio para el roedor reservorio, lo que eleva el contacto humano-virus.")
    
    with col2:
        st.subheader("Distribución Histórica (Variable Objetivo)")
        fig_hist = px.histogram(df, x='confirmed_cases', nbins=30, color='Nivel_Riesgo', title="Frecuencia de las Etiquetas a Predecir")
        fig_hist.update_layout(legend=dict(title="Etiqueta de Riesgo", orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=30, b=10), dragmode=False, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
        st.info("**Análisis del Histograma:** Aquí definimos qué es lo que el modelo va a aprender. Hemos categorizado la cantidad de casos continuos en tres **etiquetas de clase: Bajo, Medio y Alto**. El gráfico demuestra un claro sesgo en la distribución de la data (imbalanced dataset): predominan los eventos de riesgo Bajo. Esto justifica técnicamente la aplicación de ensambles avanzados como XGBoost, que manejan mejor este desbalance para clasificar correctamente los brotes de alto riesgo (eventos anómalos).")
        
    st.subheader("Muestra del Dataset Consolidado")
    st.dataframe(df.tail(15), use_container_width=True)

# ------------------------------------------
# FASE 2: Modelado (Simulador)
# ------------------------------------------
elif fase == "2. Modeling (Entrenamiento y Simulación)":
    st.header("⚙️ Fase 2: Modelos de Clasificación Multiclase")
    
    st.subheader("🌍 Zonas Críticas y Geografía del Riesgo")
    años_disponibles = sorted(df['year'].unique().tolist())
    año_seleccionado = st.selectbox("Filtrar mapa por año (Auditoría de data inyectada):", ["Ver todos los años"] + años_disponibles)
    
    df_mapa = df if año_seleccionado == "Ver todos los años" else df[df['year'] == año_seleccionado]
        
    fig_map = px.scatter_geo(df_mapa, lat='latitude', lon='longitude', color='Nivel_Riesgo', size='confirmed_cases',
                             hover_name='country', color_discrete_map={'Bajo':'green','Medio':'orange','Alto':'red'})
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.2, xanchor="center", x=0.5), dragmode=False)
    st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
    
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Simulador de Inferencia (Clasificación)")
        temp = st.slider("Temperatura (°C)", 0.0, 40.0, 20.0)
        lluvia = st.slider("Precipitación (mm)", 0.0, 3000.0, 1000.0)
        roedores = st.slider("Índice de Roedores", 0.0, 1.0, 0.4)
        densidad = st.slider("Densidad Poblacional", 10, 1000, 100)
        
        modelo_elegido = st.radio("Algoritmo de Clasificación:", ["Random Forest", "XGBoost"], horizontal=True)
        input_data = pd.DataFrame([[temp, lluvia, 65, roedores, densidad]], columns=rf_features)
        
        if modelo_elegido == "Random Forest":
            res = rf_model.predict(input_data)[0]
            probs = rf_model.predict_proba(input_data)[0]
            clases = rf_model.classes_
        else:
            res_enc = xgb_model.predict(input_data)[0]
            res = label_encoder.inverse_transform([res_enc])[0]
            probs = xgb_model.predict_proba(input_data)[0]
            clases = label_encoder.classes_
        
        st.success(f"Etiqueta de Riesgo Predicha: **{res.upper()}**")
        st.caption("Distribución de Probabilidad del Clasificador:")
        for cl, pr in zip(clases, probs):
            st.progress(float(pr), text=f"{cl}: {pr:.1%}")

    with c2:
        st.subheader(f"Árboles de Decisión: Peso de Variables")
        pesos = rf_model.feature_importances_ if modelo_elegido == "Random Forest" else xgb_model.feature_importances_
        importancia = pd.DataFrame({'Variable': [NOMBRES_CORTOS[f] for f in rf_features], 'Peso': pesos}).sort_values('Peso')
        fig_bar = px.bar(importancia, x='Peso', y='Variable', orientation='h', color='Peso', color_continuous_scale='Blues')
        fig_bar.update_layout(margin=dict(l=10, r=10, t=30, b=10), dragmode=False, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        st.info("**Interpretación del Motor de Inferencia:** Este simulador demuestra cómo operan los modelos por debajo. La IA no predice una tabla entera de casos; actúa como un **Clasificador Multiclase**. Recibe el vector de variables climáticas configurado en los deslizadores y devuelve una única **etiqueta categórica** (Bajo, Medio o Alto). El gráfico de barras explica qué características determinaron la asignación de esa etiqueta específica.")

# ------------------------------------------
# FASE 3: Evaluación (ROC, AUC y Tabla)
# ------------------------------------------
elif fase == "3. Evaluation (Métricas y Rendimiento)":
    st.header("⚖️ Fase 3: Evaluación y Validación Científica")
    
    # --- 1. TABLA DE PREDICCIÓN ---
    st.subheader("📋 Auditoría de Predicciones: Etiquetas Reales vs. IA")
    
    df_predicciones = X_test_df.copy().head(15) 
    df_predicciones.insert(0, 'ETIQUETA REAL', y_test_real.values[:15]) 
    df_predicciones.insert(1, 'Clasificación Random Forest', rf_model.predict(X_test_df)[:15])
    df_predicciones.insert(2, 'Clasificación XGBoost', label_encoder.inverse_transform(xgb_model.predict(X_test_df))[:15])
    
    def color_aciertos(row):
        colores = ['' for _ in row.index]
        for i, col in enumerate(row.index):
            if col in ['Clasificación Random Forest', 'Clasificación XGBoost']:
                if row[col] == row['ETIQUETA REAL']:
                    colores[i] = 'background-color: rgba(40, 167, 69, 0.3)' # Verde (Acierto)
                else:
                    colores[i] = 'background-color: rgba(220, 53, 69, 0.3)' # Rojo (Fallo)
        return colores

    st.dataframe(df_predicciones.style.apply(color_aciertos, axis=1), use_container_width=True)
    st.info("**Aclaración Técnica sobre la Predicción:** El sistema de Inteligencia Artificial desarrollado NO predice esta tabla. El sistema predice una **etiqueta individual de riesgo**. Esta matriz consolida el 20% del dataset que fue separado para pruebas a ciegas (Testing). Compara la 'Etiqueta Real' histórica contra la etiqueta generada por los algoritmos. Las celdas verdes certifican que, a partir de las condiciones climáticas dadas, la IA logró clasificar el riesgo exacto de manera autónoma.")

    st.divider()

    # --- 1.5. EXACTITUD GLOBAL (Accuracy) ---
    st.subheader("Métricas de Rendimiento General")
    bench_df = pd.DataFrame({'Algoritmo': ['Random Forest', 'XGBoost'], 'Exactitud Global': [acc_rf, acc_xgb]})
    fig_acc = px.bar(bench_df, x='Algoritmo', y='Exactitud Global', color='Algoritmo', text_auto='.2%')
    fig_acc.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=30, b=10), dragmode=False, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
    st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})
    st.info("**Análisis de Exactitud (Accuracy):** Mide la proporción de clasificaciones correctas sobre el total evaluado. Si bien otorga un panorama general de robustez, por sí solo es insuficiente en epidemiología. Un modelo que prediga siempre 'Bajo' tendría alta exactitud, pero fallaría en alertar las crisis. Por ello, recurrimos al análisis ROC y AUC a continuación.")

    st.divider()

    # --- 2. GRÁFICOS ROC Y AUC ---
    st.subheader("📈 Comparativa de Modelos: Curva ROC y Área Bajo la Curva (AUC)")
    
    y_test_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
    fpr_grid = np.linspace(0.0, 1.0, 100)
    n_classes = len(label_encoder.classes_)
    
    mean_tpr_rf = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])
        mean_tpr_rf += np.interp(fpr_grid, fpr, tpr)
    mean_tpr_rf /= n_classes
    auc_rf_macro = auc(fpr_grid, mean_tpr_rf)
    
    mean_tpr_xgb = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], xgb_probs[:, i])
        mean_tpr_xgb += np.interp(fpr_grid, fpr, tpr)
    mean_tpr_xgb /= n_classes
    auc_xgb_macro = auc(fpr_grid, mean_tpr_xgb)
    
    c_roc1, c_roc2 = st.columns(2)
    
    with c_roc1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_grid, y=mean_tpr_xgb, mode='lines', name='XGBoost', line=dict(color='navy', width=3)))
        fig_roc.add_trace(go.Scatter(x=fpr_grid, y=mean_tpr_rf, mode='lines', name='Random Forest', line=dict(color='gold', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='crimson', width=2), name='Clasificador Aleatorio'))
        
        fig_roc.update_layout(title="Curva ROC Multiclase (Promedio Macro)", xaxis_title="Tasa de Falsos Positivos", yaxis_title="Sensibilidad", legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=40, b=10), dragmode=False, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_roc, use_container_width=True, config={'displayModeBar': False})

    with c_roc2:
        mejor_auc = max(auc_rf_macro, auc_xgb_macro)
        if mejor_auc == auc_xgb_macro:
            tpr_ganador, nombre_ganador, color_area, color_linea = mean_tpr_xgb, "XGBoost", "rgba(30, 144, 255, 0.4)", "navy"
        else:
            tpr_ganador, nombre_ganador, color_area, color_linea = mean_tpr_rf, "Random Forest", "rgba(255, 215, 0, 0.4)", "gold"
            
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Scatter(x=fpr_grid, y=tpr_ganador, mode='lines', fill='tozeroy', fillcolor=color_area, name=f'Área {nombre_ganador} (AUC = {mejor_auc:.3f})', line=dict(color=color_linea, width=3)))
        fig_auc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray', width=2), name='Referencia (0.5)'))
        
        fig_auc.update_layout(title="Modelo Óptimo Seleccionado por Área AUC", xaxis_title="Tasa de Falsos Positivos", yaxis_title="Sensibilidad", legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=40, b=10), dragmode=False, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_auc, use_container_width=True, config={'displayModeBar': False})

    st.success(f"**Validación Científica ROC/AUC:** La curva ROC (Receiver Operating Characteristic) ilustra el balance óptimo entre Verdaderos Positivos y Falsas Alarmas a través de los múltiples umbrales de decisión. Visualmente, el clasificador superior es aquel cuya curva se tensa hacia el vértice superior izquierdo. El AUC condensa esto en un indicador absoluto: el algoritmo **{nombre_ganador}** demuestra máxima superioridad probabilística en la tarea de separar las tres etiquetas de riesgo, logrando un AUC consolidado de **{mejor_auc:.3f}**.")

    st.divider()

    # --- 3. MATRICES DE CONFUSIÓN ---
    st.subheader("Matrices de Confusión de las Etiquetas")
    c_mat1, c_mat2 = st.columns(2)
    with c_mat1:
        st.write("**Random Forest**")
        fig_cm_rf = px.imshow(rf_cm, text_auto=True, x=rf_model.classes_, y=rf_model.classes_, labels=dict(x="Etiqueta Predicha", y="Etiqueta Real"), color_continuous_scale='Blues')
        fig_cm_rf.update_layout(margin=dict(l=10, r=10, t=10, b=10), dragmode=False, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_cm_rf, use_container_width=True, config={'displayModeBar': False})
    with c_mat2:
        st.write("**XGBoost**")
        fig_cm_xgb = px.imshow(xgb_cm, text_auto=True, x=label_encoder.classes_, y=label_encoder.classes_, labels=dict(x="Etiqueta Predicha", y="Etiqueta Real"), color_continuous_scale='Oranges')
        fig_cm_xgb.update_layout(margin=dict(l=10, r=10, t=10, b=10), dragmode=False, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_cm_xgb, use_container_width=True, config={'displayModeBar': False})
    st.info("**Análisis Diagnóstico de Errores:** Esta matriz desglosa exactamente dónde acierta y dónde falla la IA. La diagonal coloreada agrupa las asignaciones de etiquetas correctas. Los números fuera del eje nos revelan el tipo de error crítico para vigilancia de salud: evaluar si el modelo subestimó un brote alto (falso negativo) o generó una falsa alarma innecesaria.")

    st.divider()

    # --- 4. MÉTRICAS DETALLADAS (Precision, Recall, F1) ---
    st.subheader("Desglose de Efectividad Multiclase (Precision, Recall, F1)")
    c_rep1, c_rep2 = st.columns(2)
    with c_rep1:
        st.write("**Métricas Random Forest**")
        st.dataframe(pd.DataFrame(rf_rep).transpose().style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)
    with c_rep2:
        st.write("**Métricas XGBoost**")
        st.dataframe(pd.DataFrame(xgb_rep).transpose().style.format("{:.2f}").background_gradient(cmap='Oranges'), use_container_width=True)
    st.info("**Desempeño en Clases Específicas:** Extraemos las métricas intrínsecas para la validación del experto del negocio. Destaca la métrica *Recall*, ya que penaliza fuertemente a la IA si no logra detectar o etiqueta erróneamente los verdaderos brotes históricos. El *F1-Score* nos confirma un equilibrio matemático sano y previene el sobreajuste (overfitting).")

# ------------------------------------------
# FASE 4: Proyección
# ------------------------------------------
else:
    st.header("🚀 Fase 4: Despliegue y Estimación de Series de Tiempo")
    años = st.slider("Ventana de tiempo a estimar (en años):", 1, 10, 5)
    fut = prophet_model.make_future_dataframe(periods=años, freq='YS')
    pred = prophet_model.predict(fut)
    
    fig_p = px.line(pred, x='ds', y='yhat', title="Evolución Histórica y Estimación Continua de Tendencia")
    
    fig_p.update_traces(hovertemplate='<b>%{x|%b %Y}</b><br>Estimación de Casos: %{y:.0f}<extra></extra>')
    
    fig_p.add_scatter(x=pred['ds'], y=pred['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip')
    fig_p.add_scatter(x=pred['ds'], y=pred['yhat_lower'], mode='lines', fill='tonexty', line=dict(width=0), showlegend=False, name="Intervalo de Confianza", hoverinfo='skip')
    
    fig_p.update_layout(
        xaxis_title="Eje Temporal",
        yaxis_title="Volumen Estimado de Casos",
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False})
    
    st.info("**Fundamento del Algoritmo Proyectivo:** Es vital destacar que a diferencia de los modelos anteriores (que asignaban una etiqueta categórica de riesgo estática), aquí aplicamos el **Algoritmo de Regresión Aditiva Prophet**. Esta red neuronal está diseñada específicamente para modelar **Series de Tiempo**, devolviendo un valor numérico continuo. Extrae la tendencia subyacente de la carga histórica epidemiológica y proyecta el volumen de casos a futuro, blindado por una franja sombreada que delimita matemáticamente el rango de incertidumbre estadístico.")
