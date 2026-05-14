import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Configuración General
st.set_page_config(page_title="Vigilancia Hantavirus IA", layout="wide", initial_sidebar_state="expanded")
st.title("🦠 Predicción de Brotes de Hantavirus")
st.markdown("*Proyecto basado en la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining)*")

# 2. Carga y Preparación de Datos (Data Preparation)
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

# --- BOTÓN DE REFRESCO DE CACHÉ ---
if st.sidebar.button("♻️ Recargar Dataset desde Disco"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# 3. Entrenamiento (Modeling)
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
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train_enc)
    xgb_pred = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test_enc, xgb_pred)
    xgb_cm = confusion_matrix(y_test_enc, xgb_pred, labels=le.transform(le.classes_))
    xgb_rep = classification_report(y_test_enc, xgb_pred, target_names=le.classes_, output_dict=True)

    # Prophet
    df_p = datos.groupby('year')['confirmed_cases'].sum().reset_index().rename(columns={'year':'ds', 'confirmed_cases':'y'})
    df_p['ds'] = pd.to_datetime(df_p['ds'], format='%Y')
    m_prophet = Prophet().fit(df_p)
    
    return rf, xgb, le, acc_rf, acc_xgb, rf_cm, xgb_cm, rf_rep, xgb_rep, m_prophet, features, X_test, y_test

rf_model, xgb_model, label_encoder, acc_rf, acc_xgb, rf_cm, xgb_cm, rf_rep, xgb_rep, prophet_model, rf_features, X_test_df, y_test_real = entrenar_modelos(df)

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

# FASE 1: Exploración de Datos
if fase == "1. Data Understanding (Exploración)":
    st.header("📊 Fase 1: Comprensión y Procesamiento de Datos")
    st.write("Análisis exploratorio de las variables climáticas y su relación con los casos de Hantavirus.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriz de Correlación Climática")
        corr_matrix = df[rf_features + ['confirmed_cases']].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', origin='lower')
        st.plotly_chart(fig_corr, width="stretch")
    
    with col2:
        st.subheader("Distribución Histórica de Casos")
        fig_hist = px.histogram(df, x='confirmed_cases', nbins=30, color='Nivel_Riesgo', title="Frecuencia de Brotes por Nivel de Riesgo")
        st.plotly_chart(fig_hist, width="stretch")
        
    st.subheader("Muestra del Dataset Consolidado")
    st.dataframe(df.tail(15), use_container_width=True)

# FASE 2: Modelado (Simulador)
elif fase == "2. Modeling (Entrenamiento y Simulación)":
    st.header("⚙️ Fase 2: Modelado Predictivo Post-Entrenamiento")
    
    st.subheader("🌍 Zonas Críticas y Geografía del Riesgo")
    
    # --- NUEVO FILTRO PARA VERIFICAR LOS DATOS INYECTADOS ---
    años_disponibles = sorted(df['year'].unique().tolist())
    año_seleccionado = st.selectbox("Filtrar mapa por año (Selecciona 2026 para ver la data inyectada):", ["Ver todos los años"] + años_disponibles)
    
    if año_seleccionado == "Ver todos los años":
        df_mapa = df
    else:
        df_mapa = df[df['year'] == año_seleccionado]
        
    fig_map = px.scatter_geo(df_mapa, lat='latitude', lon='longitude', color='Nivel_Riesgo', size='confirmed_cases',
                             hover_name='country', color_discrete_map={'Bajo':'green','Medio':'orange','Alto':'red'})
    st.plotly_chart(fig_map, width="stretch")
    # --------------------------------------------------------
    
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Simulador de Variables")
        temp = st.slider("Temperatura (°C)", 0.0, 40.0, 20.0)
        lluvia = st.slider("Precipitación (mm)", 0.0, 3000.0, 1000.0)
        roedores = st.slider("Índice de Roedores", 0.0, 1.0, 0.4)
        densidad = st.slider("Densidad Poblacional", 10, 1000, 100)
        
        modelo_elegido = st.radio("Motor de Inferencia:", ["Random Forest", "XGBoost"], horizontal=True)
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
        
        st.success(f"Nivel de Riesgo Calculado: **{res.upper()}**")
        
        for cl, pr in zip(clases, probs):
            st.progress(float(pr), text=f"{cl}: {pr:.1%}")

    with c2:
        st.subheader(f"Importancia de Variables ({modelo_elegido})")
        if modelo_elegido == "Random Forest":
            pesos = rf_model.feature_importances_
        else:
            pesos = xgb_model.feature_importances_
            
        importancia = pd.DataFrame({'Variable': rf_features, 'Peso': pesos}).sort_values('Peso')
        fig_bar = px.bar(importancia, x='Peso', y='Variable', orientation='h', color='Peso', color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, width="stretch")

# FASE 3: Evaluación de Métricas
elif fase == "3. Evaluation (Métricas y Rendimiento)":
    st.header("⚖️ Fase 3: Evaluación y Tablas de Métricas")
    
    st.subheader("📋 Resultados de Predicción (Ground Truth vs Machine Learning)")
    st.write("Esta tabla muestra los datos separados para evaluación (20% del dataset). Compara el riesgo real que ocurrió frente a lo que predijeron nuestros modelos entrenados.")
    
    df_predicciones = X_test_df.copy()
    df_predicciones.insert(0, 'RIESGO REAL', y_test_real.values) 
    df_predicciones.insert(1, 'Predicción Random Forest', rf_model.predict(X_test_df))
    df_predicciones.insert(2, 'Predicción XGBoost', label_encoder.inverse_transform(xgb_model.predict(X_test_df)))
    
    def color_aciertos(row):
        colores = ['' for _ in row.index]
        for i, col in enumerate(row.index):
            if col in ['Predicción Random Forest', 'Predicción XGBoost']:
                if row[col] == row['RIESGO REAL']:
                    colores[i] = 'background-color: rgba(40, 167, 69, 0.3)' 
                else:
                    colores[i] = 'background-color: rgba(220, 53, 69, 0.3)' 
        return colores

    st.dataframe(df_predicciones.style.apply(color_aciertos, axis=1), use_container_width=True)
    st.divider()

    st.subheader("Tabla Resumen de Exactitud (Accuracy)")
    bench_df = pd.DataFrame({'Algoritmo': ['Random Forest', 'XGBoost'], 'Exactitud Global': [acc_rf, acc_xgb]})
    st.plotly_chart(px.bar(bench_df, x='Algoritmo', y='Exactitud Global', color='Algoritmo', text_auto='.2%'), width="stretch")
    
    st.divider()
    st.subheader("Matrices de Confusión")
    c_mat1, c_mat2 = st.columns(2)
    with c_mat1:
        st.write("**Random Forest**")
        fig_cm_rf = px.imshow(rf_cm, text_auto=True, x=rf_model.classes_, y=rf_model.classes_, labels=dict(x="Predicción", y="Realidad"), color_continuous_scale='Blues')
        st.plotly_chart(fig_cm_rf, width="stretch")
    with c_mat2:
        st.write("**XGBoost**")
        fig_cm_xgb = px.imshow(xgb_cm, text_auto=True, x=label_encoder.classes_, y=label_encoder.classes_, labels=dict(x="Predicción", y="Realidad"), color_continuous_scale='Oranges')
        st.plotly_chart(fig_cm_xgb, width="stretch")

    st.divider()
    st.subheader("Tabla de Métricas Detalladas (Precision, Recall, F1-Score)")
    c_rep1, c_rep2 = st.columns(2)
    with c_rep1:
        st.write("**Métricas Random Forest**")
        st.dataframe(pd.DataFrame(rf_rep).transpose().style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)
    with c_rep2:
        st.write("**Métricas XGBoost**")
        st.dataframe(pd.DataFrame(xgb_rep).transpose().style.format("{:.2f}").background_gradient(cmap='Oranges'), use_container_width=True)

# FASE 4: Proyección
else:
    st.header("🚀 Fase 4: Despliegue y Proyección (Prophet)")
    años = st.slider("Años a proyectar en el futuro:", 1, 10, 5)
    fut = prophet_model.make_future_dataframe(periods=años, freq='YS')
    pred = prophet_model.predict(fut)
    
    fig_p = px.line(pred, x='ds', y='yhat', title="Curva de Casos Históricos y Predicción de Tendencia")
    fig_p.add_scatter(x=pred['ds'], y=pred['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False)
    fig_p.add_scatter(x=pred['ds'], y=pred['yhat_lower'], mode='lines', fill='tonexty', line=dict(width=0), showlegend=False, name="Margen de Error")
    st.plotly_chart(fig_p, width="stretch")
    
    st.write("Esta proyección utiliza el algoritmo aditivo Prophet para estimar la carga epidemiológica futura basada en estacionalidad.")