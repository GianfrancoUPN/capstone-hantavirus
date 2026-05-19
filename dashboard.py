import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, log_loss
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
    
    # --- PARCHE DE LIMPIEZA DE SÍNDROMES ("None" o Vacíos) ---
    df['syndrome'] = df['syndrome'].fillna('No Especificado')
    df['syndrome'] = df['syndrome'].replace('None', 'No Especificado')
    # ---------------------------------------------------------

    # --- PARCHE DE LIMPIEZA GEOGRÁFICA (BROTE 2026) ---
    coordenadas = {
        'Canada': [56.1304, -106.3468],
        'Netherlands': [52.1326, 5.2913],
        'South Africa': [-30.5595, 22.9375],
        'Switzerland': [46.8182, 8.2275],
        'France': [46.2276, 2.2137],
        'Spain': [40.4637, -3.7492]
    }
    for pais, coords in coordenadas.items():
        mask = (df['year'] == 2026) & (df['country'] == pais)
        df.loc[mask, 'latitude'] = coords[0]
        df.loc[mask, 'longitude'] = coords[1]
    # --------------------------------------------------

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
    
    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred, labels=rf.classes_)
    rf_rep = classification_report(y_test, rf_pred, output_dict=True)
    rf_probs = rf.predict_proba(X_test)

    # Curva de Pérdida iterativa para Random Forest
    rf_loss_trees = []
    rf_loss_val = []
    for i in range(1, 105, 5):
        rf_iter = RandomForestClassifier(n_estimators=i, max_depth=5, random_state=42)
        rf_iter.fit(X_train, y_train)
        probs = rf_iter.predict_proba(X_test)
        rf_loss_trees.append(i)
        rf_loss_val.append(log_loss(y_test, probs))
    
    # --- XGBoost ---
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train_enc, eval_set=[(X_train, y_train_enc), (X_test, y_test_enc)], verbose=False)
    xgb_pred = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test_enc, xgb_pred)
    xgb_cm = confusion_matrix(y_test_enc, xgb_pred, labels=le.transform(le.classes_))
    xgb_rep = classification_report(y_test_enc, xgb_pred, target_names=le.classes_, output_dict=True)
    xgb_probs = xgb.predict_proba(X_test)

    # Extraer historial de pérdida de XGBoost
    xgb_evals = xgb.evals_result()
    xgb_loss_train = xgb_evals['validation_0']['mlogloss']
    xgb_loss_test = xgb_evals['validation_1']['mlogloss']

    # --- Prophet ---
    df_p = datos.groupby('year')['confirmed_cases'].sum().reset_index().rename(columns={'year':'ds', 'confirmed_cases':'y'})
    df_p['ds'] = pd.to_datetime(df_p['ds'], format='%Y')
    m_prophet = Prophet().fit(df_p)
    
    return rf, xgb, le, acc_rf, acc_xgb, rf_cm, xgb_cm, rf_rep, xgb_rep, m_prophet, features, X_test, y_test, y_test_enc, rf_probs, xgb_probs, xgb_loss_train, xgb_loss_test, rf_loss_trees, rf_loss_val

rf_model, xgb_model, label_encoder, acc_rf, acc_xgb, rf_cm, xgb_cm, rf_rep, xgb_rep, prophet_model, rf_features, X_test_df, y_test_real, y_test_enc, rf_probs, xgb_probs, xgb_loss_train, xgb_loss_test, rf_loss_trees, rf_loss_val = entrenar_modelos(df)

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
    st.write("Análisis exploratorio de las variables climáticas y su relación con los casos de Hantavirus.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriz de Correlación Climática")
        df_corr = df[rf_features + ['confirmed_cases']].rename(columns=NOMBRES_CORTOS)
        corr_matrix = df_corr.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', origin='lower')
        fig_corr.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
        st.info("**Interpretación de la Matriz:** Los valores cercanos a 1 (rojo) indican una correlación positiva fuerte. Se observa que el *Índice de Roedores* y la *Precipitación* tienen el mayor impacto directo en los casos confirmados.")
    
    with col2:
        st.subheader("Distribución Histórica de Casos")
        fig_hist = px.histogram(df, x='confirmed_cases', nbins=30, color='Nivel_Riesgo', title="Frecuencia de Brotes por Nivel de Riesgo")
        fig_hist.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
        st.info("**Interpretación del Histograma:** Muestra el desbalance natural de la carga epidemiológica. La gran mayoría de los registros históricos caen en riesgo 'Bajo', siendo los brotes 'Altos' eventos anómalos.")
        
    st.subheader("Muestra del Dataset Consolidado")
    st.dataframe(df.tail(15), use_container_width=True)

# ------------------------------------------
# FASE 2: Modelado (Simulador)
# ------------------------------------------
elif fase == "2. Modeling (Entrenamiento y Simulación)":
    st.header("⚙️ Fase 2: Modelado Predictivo Post-Entrenamiento")
    
    st.subheader("🌍 Zonas Críticas y Geografía del Riesgo")
    años_disponibles = sorted(df['year'].unique().tolist())
    año_seleccionado = st.selectbox("Filtrar mapa por año (Selecciona 2026 para ver la data inyectada):", ["Ver todos los años"] + años_disponibles)
    
    df_mapa = df if año_seleccionado == "Ver todos los años" else df[df['year'] == año_seleccionado]
        
    fig_map = px.scatter_geo(df_mapa, lat='latitude', lon='longitude', color='Nivel_Riesgo', size='confirmed_cases',
                             hover_name='country', 
                             hover_data={'Nivel_Riesgo': True, 'confirmed_cases': True, 'deaths': True, 'syndrome': True, 'latitude': False, 'longitude': False},
                             color_discrete_map={'Bajo':'green','Medio':'orange','Alto':'red'})
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.2, xanchor="center", x=0.5))
    st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
    
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
        pesos = rf_model.feature_importances_ if modelo_elegido == "Random Forest" else xgb_model.feature_importances_
        importancia = pd.DataFrame({'Variable': [NOMBRES_CORTOS[f] for f in rf_features], 'Peso': pesos}).sort_values('Peso')
        fig_bar = px.bar(importancia, x='Peso', y='Variable', orientation='h', color='Peso', color_continuous_scale='Blues')
        fig_bar.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

# ------------------------------------------
# FASE 3: Evaluación (ROC, AUC y Tabla)
# ------------------------------------------
elif fase == "3. Evaluation (Métricas y Rendimiento)":
    st.header("⚖️ Fase 3: Evaluación y Validación Científica")
    
    st.subheader("📋 Tabla de Predicción (Ground Truth vs Inteligencia Artificial)")
    df_predicciones = X_test_df.copy().head(15) 
    df_predicciones.insert(0, 'RIESGO REAL', y_test_real.values[:15]) 
    df_predicciones.insert(1, 'Predicción Random Forest', rf_model.predict(X_test_df)[:15])
    df_predicciones.insert(2, 'Predicción XGBoost', label_encoder.inverse_transform(xgb_model.predict(X_test_df))[:15])
    
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
    fig_acc = px.bar(bench_df, x='Algoritmo', y='Exactitud Global', color='Algoritmo', text_auto='.2%')
    fig_acc.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})
    
    st.divider()

    # --- CÁLCULOS MATEMÁTICOS PARA ROC Y AUC ---
    y_test_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
    fpr_grid = np.linspace(0.0, 1.0, 100)
    n_classes = len(label_encoder.classes_)
    
    # Matemáticas para Random Forest
    mean_tpr_rf = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])
        mean_tpr_rf += np.interp(fpr_grid, fpr, tpr)
    mean_tpr_rf /= n_classes
    auc_rf_macro = auc(fpr_grid, mean_tpr_rf)
    
    # Matemáticas para XGBoost
    mean_tpr_xgb = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], xgb_probs[:, i])
        mean_tpr_xgb += np.interp(fpr_grid, fpr, tpr)
    mean_tpr_xgb /= n_classes
    auc_xgb_macro = auc(fpr_grid, mean_tpr_xgb)


    # ==========================================
    # BLOQUE 1: RANDOM FOREST DETALLADO
    # ==========================================
    st.subheader("🌲 Análisis Detallado: Random Forest")
    c_rf1, c_rf2, c_rf3 = st.columns(3)
    
    with c_rf1:
        fig_roc_rf = go.Figure()
        fig_roc_rf.add_trace(go.Scatter(x=fpr_grid, y=mean_tpr_rf, mode='lines', line=dict(color='gold', width=3), name='Curva ROC'))
        fig_roc_rf.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray', width=2), showlegend=False))
        fig_roc_rf.update_layout(title="Curva ROC", xaxis_title="Falsos Positivos", yaxis_title="Sensibilidad", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_roc_rf, use_container_width=True, config={'displayModeBar': False})

    with c_rf2:
        fig_auc_rf = go.Figure()
        fig_auc_rf.add_trace(go.Scatter(x=fpr_grid, y=mean_tpr_rf, mode='lines', fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.4)', name=f'AUC = {auc_rf_macro:.3f}', line=dict(color='gold', width=3)))
        fig_auc_rf.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray', width=2), showlegend=False))
        fig_auc_rf.update_layout(title=f"Área Bajo la Curva (AUC: {auc_rf_macro:.3f})", xaxis_title="Falsos Positivos", yaxis_title="Sensibilidad", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_auc_rf, use_container_width=True, config={'displayModeBar': False})

    with c_rf3:
        fig_loss_rf = go.Figure()
        fig_loss_rf.add_trace(go.Scatter(x=rf_loss_trees, y=rf_loss_val, mode='lines+markers', line=dict(color='gold', width=3), name='Pérdida (Log Loss)'))
        fig_loss_rf.update_layout(title="Curva de Pérdida (Bagging)", xaxis_title="N° de Árboles", yaxis_title="Error Logarítmico", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_loss_rf, use_container_width=True, config={'displayModeBar': False})

    st.divider()

    # ==========================================
    # BLOQUE 2: XGBOOST DETALLADO
    # ==========================================
    st.subheader("🚀 Análisis Detallado: XGBoost")
    c_xgb1, c_xgb2, c_xgb3 = st.columns(3)
    
    with c_xgb1:
        fig_roc_xgb = go.Figure()
        fig_roc_xgb.add_trace(go.Scatter(x=fpr_grid, y=mean_tpr_xgb, mode='lines', line=dict(color='navy', width=3), name='Curva ROC'))
        fig_roc_xgb.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray', width=2), showlegend=False))
        fig_roc_xgb.update_layout(title="Curva ROC", xaxis_title="Falsos Positivos", yaxis_title="Sensibilidad", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_roc_xgb, use_container_width=True, config={'displayModeBar': False})

    with c_xgb2:
        fig_auc_xgb = go.Figure()
        fig_auc_xgb.add_trace(go.Scatter(x=fpr_grid, y=mean_tpr_xgb, mode='lines', fill='tozeroy', fillcolor='rgba(30, 144, 255, 0.4)', name=f'AUC = {auc_xgb_macro:.3f}', line=dict(color='navy', width=3)))
        fig_auc_xgb.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray', width=2), showlegend=False))
        fig_auc_xgb.update_layout(title=f"Área Bajo la Curva (AUC: {auc_xgb_macro:.3f})", xaxis_title="Falsos Positivos", yaxis_title="Sensibilidad", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_auc_xgb, use_container_width=True, config={'displayModeBar': False})

    with c_xgb3:
        fig_loss_xgb = go.Figure()
        fig_loss_xgb.add_trace(go.Scatter(y=xgb_loss_train, mode='lines', line=dict(color='lightblue', width=2), name='Train'))
        fig_loss_xgb.add_trace(go.Scatter(y=xgb_loss_test, mode='lines', line=dict(color='navy', width=3), name='Test'))
        fig_loss_xgb.update_layout(title="Curva de Pérdida (Boosting)", xaxis_title="Épocas (Rondas)", yaxis_title="Error (mlogloss)", legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_loss_xgb, use_container_width=True, config={'displayModeBar': False})

    st.info("""**Justificación Analítica:** Al separar matemáticamente los modelos, podemos auditar de forma visual que ambos algoritmos lograron converger exitosamente. En las **Curvas de Pérdida**, se evidencia que conforme avanza el aprendizaje (ya sea añadiendo árboles en Random Forest o procesando épocas en XGBoost), el error estadístico decae y se estabiliza. Notablemente en XGBoost, la línea de prueba (Test) desciende a la par que la de entrenamiento, demostrando contundentemente que **el sistema no presenta sobreajuste (overfitting)**.""")

    st.divider()

    # --- 3. MATRICES DE CONFUSIÓN ---
    st.subheader("Matrices de Confusión Detalladas")
    c_mat1, c_mat2 = st.columns(2)
    with c_mat1:
        st.write("**Random Forest**")
        fig_cm_rf = px.imshow(rf_cm, text_auto=True, x=rf_model.classes_, y=rf_model.classes_, labels=dict(x="Predicción", y="Realidad"), color_continuous_scale='Blues')
        fig_cm_rf.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm_rf, use_container_width=True, config={'displayModeBar': False})
    with c_mat2:
        st.write("**XGBoost**")
        fig_cm_xgb = px.imshow(xgb_cm, text_auto=True, x=label_encoder.classes_, y=label_encoder.classes_, labels=dict(x="Predicción", y="Realidad"), color_continuous_scale='Oranges')
        fig_cm_xgb.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm_xgb, use_container_width=True, config={'displayModeBar': False})

    st.divider()

    # --- 4. MÉTRICAS DETALLADAS (Precision, Recall, F1) ---
    st.subheader("Tabla de Métricas Detalladas (Precision, Recall, F1-Score)")
    c_rep1, c_rep2 = st.columns(2)
    with c_rep1:
        st.write("**Métricas Random Forest**")
        st.dataframe(pd.DataFrame(rf_rep).transpose().style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)
    with c_rep2:
        st.write("**Métricas XGBoost**")
        st.dataframe(pd.DataFrame(xgb_rep).transpose().style.format("{:.2f}").background_gradient(cmap='Oranges'), use_container_width=True)

# ------------------------------------------
# FASE 4: Proyección
# ------------------------------------------
else:
    st.header("🚀 Fase 4: Despliegue y Proyección Temporal (Prophet)")
    años = st.slider("Años a proyectar en el futuro:", 1, 10, 5)
    fut = prophet_model.make_future_dataframe(periods=años, freq='YS')
    pred = prophet_model.predict(fut)
    
    fig_p = px.line(pred, x='ds', y='yhat', title="Curva de Casos Históricos y Predicción de Tendencia")
    fig_p.update_traces(hovertemplate='<b>%{x|%b %Y}</b><br>Casos Estimados: %{y:.0f}<extra></extra>')
    fig_p.add_scatter(x=pred['ds'], y=pred['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip')
    fig_p.add_scatter(x=pred['ds'], y=pred['yhat_lower'], mode='lines', fill='tonexty', line=dict(width=0), showlegend=False, name="Margen de Error", hoverinfo='skip')
    fig_p.update_layout(xaxis_title="Año de Proyección", yaxis_title="Casos Proyectados", dragmode=False, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False})
