"""
Contador Virtual - POC
ML classifier (Random Forest) para clasificación fiscal de clientes.
Reduce labor manual al automatizar la categorización fiscal.

Run: streamlit run app.py
Requirements: streamlit, scikit-learn, pandas, numpy, plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
page_title="Contador Virtual",
page_icon="🧾",
layout="wide",
)

REGIMEN_LABELS = [
"Régimen Simplificado de Confianza (RESICO)",
"Régimen General de Ley (Actividad Empresarial)",
"Sueldos y Salarios",
"Régimen de Incorporación Fiscal (RIF)",
"Persona Moral - Régimen General",
]

SECTOR_OPTIONS = [
"Comercio",
"Servicios Profesionales",
"Manufactura",
"Tecnología",
"Construcción",
"Alimentos y Bebidas",
"Transporte",
"Salud",
]

# ── Dummy data generation ────────────────────────────────────────────────────

@st.cache_data
def generate_training_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
"""Generate synthetic fiscal data for training."""
rng = np.random.default_rng(seed)


records = []
for _ in range(n_samples):
    # Pick a target regime first, then generate features that correlate
    regime = rng.choice(REGIMEN_LABELS, p=[0.25, 0.20, 0.25, 0.15, 0.15])

    if regime == "Sueldos y Salarios":
        ingreso_mensual = rng.uniform(8_000, 80_000)
        num_facturas_mes = rng.integers(0, 5)
        gastos_deducibles = rng.uniform(0, ingreso_mensual * 0.15)
        tiene_empleados = 0
        es_persona_moral = 0
        antiguedad_fiscal = rng.integers(0, 30)
        num_actividades = 1
        ingresos_anuales = ingreso_mensual * 12 * rng.uniform(0.9, 1.1)

    elif regime == "Régimen Simplificado de Confianza (RESICO)":
        ingreso_mensual = rng.uniform(5_000, 291_666)  # < 3.5M annual
        num_facturas_mes = rng.integers(1, 50)
        gastos_deducibles = rng.uniform(0, ingreso_mensual * 0.10)
        tiene_empleados = int(rng.random() < 0.2)
        es_persona_moral = 0
        antiguedad_fiscal = rng.integers(0, 10)
        num_actividades = rng.integers(1, 3)
        ingresos_anuales = ingreso_mensual * 12 * rng.uniform(0.85, 1.15)

    elif regime == "Régimen General de Ley (Actividad Empresarial)":
        ingreso_mensual = rng.uniform(50_000, 500_000)
        num_facturas_mes = rng.integers(10, 200)
        gastos_deducibles = rng.uniform(ingreso_mensual * 0.2, ingreso_mensual * 0.6)
        tiene_empleados = int(rng.random() < 0.6)
        es_persona_moral = 0
        antiguedad_fiscal = rng.integers(2, 25)
        num_actividades = rng.integers(1, 5)
        ingresos_anuales = ingreso_mensual * 12 * rng.uniform(0.8, 1.2)

    elif regime == "Régimen de Incorporación Fiscal (RIF)":
        ingreso_mensual = rng.uniform(3_000, 166_666)  # < 2M annual
        num_facturas_mes = rng.integers(1, 30)
        gastos_deducibles = rng.uniform(0, ingreso_mensual * 0.20)
        tiene_empleados = int(rng.random() < 0.15)
        es_persona_moral = 0
        antiguedad_fiscal = rng.integers(0, 10)
        num_actividades = rng.integers(1, 2)
        ingresos_anuales = ingreso_mensual * 12 * rng.uniform(0.85, 1.15)

    else:  # Persona Moral
        ingreso_mensual = rng.uniform(100_000, 5_000_000)
        num_facturas_mes = rng.integers(20, 500)
        gastos_deducibles = rng.uniform(ingreso_mensual * 0.3, ingreso_mensual * 0.7)
        tiene_empleados = 1
        es_persona_moral = 1
        antiguedad_fiscal = rng.integers(1, 30)
        num_actividades = rng.integers(1, 8)
        ingresos_anuales = ingreso_mensual * 12 * rng.uniform(0.75, 1.25)

    sector = rng.choice(SECTOR_OPTIONS)

    records.append({
        "ingreso_mensual_promedio": round(ingreso_mensual, 2),
        "ingresos_anuales": round(ingresos_anuales, 2),
        "num_facturas_mes": int(num_facturas_mes),
        "gastos_deducibles_mes": round(gastos_deducibles, 2),
        "tiene_empleados": tiene_empleados,
        "es_persona_moral": es_persona_moral,
        "antiguedad_fiscal_anios": int(antiguedad_fiscal),
        "num_actividades_economicas": int(num_actividades),
        "sector": sector,
        "regimen_fiscal": regime,
    })

return pd.DataFrame(records)


# ── Model training ───────────────────────────────────────────────────────────

FEATURE_COLS = [
"ingreso_mensual_promedio",
"ingresos_anuales",
"num_facturas_mes",
"gastos_deducibles_mes",
"tiene_empleados",
"es_persona_moral",
"antiguedad_fiscal_anios",
"num_actividades_economicas",
"sector_encoded",
]

@st.cache_resource
def train_model(df: pd.DataFrame):
"""Train a Random Forest classifier and return model + artifacts."""
le_sector = LabelEncoder()
df = df.copy()
df["sector_encoded"] = le_sector.fit_transform(df[“sector”])


X = df[FEATURE_COLS]
y = df["regimen_fiscal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

return clf, le_sector, accuracy, report, cm, X_test, y_test


# ── UI ───────────────────────────────────────────────────────────────────────

def main():
st.title("🧾 Contador Virtual")
st.markdown(
"**POC** — Clasificador ML que sugiere el régimen fiscal óptimo "
"para cada cliente, reduciendo trabajo manual del contador."
)


# Load data & model
df = generate_training_data()
clf, le_sector, accuracy, report, cm, X_test, y_test = train_model(df)

tab_predict, tab_model, tab_data = st.tabs(
    ["🔮 Predicción", "📊 Modelo", "🗂️ Datos de entrenamiento"]
)

# ── Tab: Prediction ──────────────────────────────────────────────────
with tab_predict:
    st.subheader("Ingresa los datos fiscales del cliente")

    col1, col2 = st.columns(2)

    with col1:
        ingreso_mensual = st.number_input(
            "Ingreso mensual promedio (MXN)",
            min_value=0.0, max_value=10_000_000.0,
            value=50_000.0, step=1_000.0, format="%.2f",
        )
        ingresos_anuales = st.number_input(
            "Ingresos anuales estimados (MXN)",
            min_value=0.0, max_value=100_000_000.0,
            value=ingreso_mensual * 12, step=10_000.0, format="%.2f",
        )
        num_facturas = st.slider("Facturas emitidas por mes", 0, 500, 15)
        gastos_deducibles = st.number_input(
            "Gastos deducibles mensuales (MXN)",
            min_value=0.0, max_value=10_000_000.0,
            value=10_000.0, step=500.0, format="%.2f",
        )

    with col2:
        sector = st.selectbox("Sector económico", SECTOR_OPTIONS)
        tiene_empleados = st.radio(
            "¿Tiene empleados?", ["No", "Sí"], horizontal=True
        )
        es_persona_moral = st.radio(
            "¿Es persona moral?", ["No", "Sí"], horizontal=True
        )
        antiguedad = st.slider("Antigüedad fiscal (años)", 0, 40, 3)
        num_actividades = st.slider("Número de actividades económicas", 1, 10, 1)

    if st.button("🚀 Clasificar cliente", type="primary", use_container_width=True):
        sector_enc = le_sector.transform([sector])[0]
        input_data = pd.DataFrame([{
            "ingreso_mensual_promedio": ingreso_mensual,
            "ingresos_anuales": ingresos_anuales,
            "num_facturas_mes": num_facturas,
            "gastos_deducibles_mes": gastos_deducibles,
            "tiene_empleados": 1 if tiene_empleados == "Sí" else 0,
            "es_persona_moral": 1 if es_persona_moral == "Sí" else 0,
            "antiguedad_fiscal_anios": antiguedad,
            "num_actividades_economicas": num_actividades,
            "sector_encoded": sector_enc,
        }])

        proba = clf.predict_proba(input_data)[0]
        pred_idx = np.argmax(proba)
        pred_label = clf.classes_[pred_idx]
        confidence = proba[pred_idx]

        st.divider()
        st.subheader("Resultado")

        # Confidence color
        if confidence >= 0.75:
            color = "green"
            emoji = "✅"
        elif confidence >= 0.50:
            color = "orange"
            emoji = "⚠️"
        else:
            color = "red"
            emoji = "🔴"

        res_col1, res_col2 = st.columns([2, 1])
        with res_col1:
            st.markdown(
                f"### {emoji} {pred_label}\n"
                f"**Confianza:** :{color}[**{confidence:.1%}**]"
            )
        with res_col2:
            st.metric("Confianza", f"{confidence:.1%}")

        # Probability breakdown
        st.markdown("#### Distribución de probabilidades")
        proba_df = (
            pd.DataFrame({"Régimen": clf.classes_, "Probabilidad": proba})
            .sort_values("Probabilidad", ascending=True)
        )
        fig = px.bar(
            proba_df, x="Probabilidad", y="Régimen",
            orientation="h",
            color="Probabilidad",
            color_continuous_scale="Tealgrn",
            text=proba_df["Probabilidad"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(
            height=300, showlegend=False,
            xaxis_tickformat=".0%",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        if confidence < 0.60:
            st.warning(
                "⚠️ La confianza es baja. Se recomienda revisión manual "
                "por un contador para este cliente."
            )

# ── Tab: Model performance ───────────────────────────────────────────
with tab_model:
    st.subheader("Desempeño del modelo")

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{accuracy:.1%}")
    m2.metric("Muestras de entrenamiento", f"{len(df) * 0.8:,.0f}")
    m3.metric("Muestras de prueba", f"{len(df) * 0.2:,.0f}")

    st.markdown("#### Matriz de confusión")
    fig_cm = px.imshow(
        cm,
        x=[c[:20] + "…" if len(c) > 20 else c for c in clf.classes_],
        y=[c[:20] + "…" if len(c) > 20 else c for c in clf.classes_],
        color_continuous_scale="Blues",
        text_auto=True,
        labels=dict(x="Predicho", y="Real", color="Conteo"),
    )
    fig_cm.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### Feature Importance")
    importance_df = (
        pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": clf.feature_importances_,
        })
        .sort_values("Importance", ascending=True)
    )
    fig_imp = px.bar(
        importance_df, x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Viridis",
    )
    fig_imp.update_layout(
        height=350, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### Classification Report")
    report_df = (
        pd.DataFrame(report)
        .T.drop(["accuracy"], errors="ignore")
        .round(3)
    )
    st.dataframe(report_df, use_container_width=True)

# ── Tab: Training data ───────────────────────────────────────────────
with tab_data:
    st.subheader("Datos sintéticos de entrenamiento")
    st.caption(f"{len(df):,} registros generados para el POC.")

    st.markdown("#### Distribución por régimen fiscal")
    dist = df["regimen_fiscal"].value_counts().reset_index()
    dist.columns = ["Régimen", "Conteo"]
    fig_dist = px.pie(dist, names="Régimen", values="Conteo", hole=0.4)
    fig_dist.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=30))
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### Ingresos por régimen")
    fig_box = px.box(
        df, x="regimen_fiscal", y="ingreso_mensual_promedio",
        color="regimen_fiscal",
        labels={
            "regimen_fiscal": "Régimen",
            "ingreso_mensual_promedio": "Ingreso mensual (MXN)",
        },
    )
    fig_box.update_layout(
        height=400, showlegend=False,
        xaxis_tickangle=-30,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### Muestra de datos")
    st.dataframe(df.head(100), use_container_width=True)


if __name__ == "__main__":
    main()
