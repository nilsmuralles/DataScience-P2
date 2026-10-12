"""App de Streamlit: explorar datos, clasificar frames y comparar modelos.

Correr desde la raíz del repo:
    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import silhouette_score

from core import (CLASS_COLORS, CLASS_LABELS, CLASSES, FEATURE_LABELS, FEATURES,
                  FIG_DIR, ROOT, load_cnn_pca, load_frame_metrics, load_models, load_results,
                  model_color_map, model_name, predict_all, sort_models)
from src.features import decode_image

st.set_page_config(page_title="Detección de eventos · Bundesliga", page_icon="⚽", layout="wide")

SURFACE = "#1a1a19"  # separa barras y puntos con el color del fondo
PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13),
              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
              hoverlabel=dict(font_size=13))


def show(fig, height=380):
    # El título va como texto de Streamlit: dentro de Plotly choca con la leyenda.
    title = fig.layout.title.text
    if title:
        st.markdown(f"**{title}**")
    fig.update_layout(height=height, title_text="", **LAYOUT)
    for ax in ("yaxis", "xaxis"):
        rng = fig.layout[ax].range
        if rng is not None:
            fig.layout[ax].autorange = False
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)


def class_order(present) -> list:
    return [c for c in CLASSES if c in set(present)]


# ------------------------------------------------------------------ header
st.title("⚽ Detección de eventos en partidos de la Bundesliga")
st.caption("Clasifica un frame de video en **bg** (sin evento), **play** (pase), "
           "**challenge** (disputa) o **throwin** (saque de banda).")

fm = load_frame_metrics()
models = load_models()
metrics = load_results("metrics")
preds = load_results("predictions")

tab_exp, tab_clf, tab_perf = st.tabs(["🔎 Explorar datos", "🎯 Clasificar", "📊 Rendimiento"])

# ============================================================ EXPLORAR
with tab_exp:
    st.subheader("Los datos con los que se entrenaron los modelos")
    st.markdown(
        f"**{len(fm):,} frames** muestreados de **{fm.video_id.nunique()} videos** "
        "(cada video es una mitad de partido). De cada frame se calcularon 5 propiedades "
        "de imagen, que son la entrada de los modelos clásicos.")

    c1, c2, c3 = st.columns([2, 2, 3])
    sel_cls = c1.multiselect("Clases", CLASSES, default=CLASSES,
                             format_func=lambda c: CLASS_LABELS[c])
    videos = sorted(fm.video_id.unique())
    sel_vid = c2.multiselect("Videos", videos, default=videos)
    feat = c3.selectbox("Propiedad", FEATURES, format_func=lambda f: FEATURE_LABELS[f])

    d = fm[fm.label.isin(sel_cls) & fm.video_id.isin(sel_vid)]
    if d.empty:
        st.info("No hay frames con esos filtros.")
    else:
        order = class_order(d.label)
        cmap = {c: CLASS_COLORS[c] for c in order}

        # --- conteo de clases
        counts = d.label.value_counts().reindex(order).rename_axis("clase").reset_index(name="frames")
        counts["pct"] = (counts.frames / counts.frames.sum() * 100).round(1)
        top = counts.sort_values("frames", ascending=False).iloc[0]
        k = st.columns(len(order))
        for col, r in zip(k, counts.itertuples(index=False)):
            col.metric(CLASS_LABELS[r.clase], f"{r.frames:,}", f"{r.pct:.1f}% del total", delta_color="off", delta_arrow="off")

        a, b = st.columns(2)
        with a:
            fig = px.bar(counts, x="clase", y="frames", color="clase", color_discrete_map=cmap,
                         text="frames", title="Frames por clase",
                         category_orders={"clase": order})
            fig.update_traces(textposition="outside", marker_line_width=0, showlegend=False,
                              hovertemplate="%{x}: %{y:,} frames<extra></extra>")
            fig.update_yaxes(title=None)
            show(fig)
        with b:
            pv = (d.groupby(["video_id", "label"]).size().rename("frames").reset_index())
            pv["%"] = pv.frames / pv.groupby("video_id").frames.transform("sum") * 100
            fig = px.bar(pv, y="video_id", x="%", color="label", orientation="h",
                         color_discrete_map=cmap, category_orders={"label": order},
                         title="Composición de clases por video", custom_data=["frames"])
            fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                              hovertemplate="%{y} · %{fullData.name}: %{x:.1f}% (%{customdata[0]:,})<extra></extra>")
            fig.update_xaxes(title="% de frames del video", range=[0, 100])
            fig.update_yaxes(title=None)
            show(fig)
        st.info(f"**Desbalance fuerte:** *{top.clase}* concentra el {top.pct:.0f}% de los frames. "
                "Un modelo que siempre responda esa clase ya tiene un accuracy alto, por eso en "
                "**Rendimiento** se compara con F1 macro (promedio del F1 de cada clase).")

        # --- distribución de una propiedad
        a, b = st.columns(2)
        with a:
            fig = px.histogram(d, x=feat, color="label", color_discrete_map=cmap, nbins=50,
                               barmode="overlay", opacity=0.6, histnorm="percent",
                               category_orders={"label": order},
                               title=f"Distribución de {FEATURE_LABELS[feat].lower()}")
            fig.update_yaxes(title="% de frames de la clase")
            fig.update_xaxes(title=FEATURE_LABELS[feat])
            show(fig)
        with b:
            fig = px.box(d, x="label", y=feat, color="label", color_discrete_map=cmap,
                         category_orders={"label": order}, points=False,
                         title=f"{FEATURE_LABELS[feat]} por clase")
            fig.update_traces(showlegend=False)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=FEATURE_LABELS[feat])
            show(fig)

        med = d.groupby("label")[feat].median().reindex(order)
        spread = (med.max() - med.min()) / d[feat].std() if d[feat].std() else 0
        st.caption(f"Las medianas por clase de *{feat}* difieren en {spread:.2f} desviaciones estándar. "
                   "Las cajas se traslapan casi por completo: una propiedad global del frame "
                   "distingue poco entre eventos, lo que explica el techo de los modelos clásicos.")

        with st.expander("Relación entre dos propiedades"):
            x2 = st.selectbox("Eje Y", [f for f in FEATURES if f != feat],
                              format_func=lambda f: FEATURE_LABELS[f])
            sample = d.sample(min(len(d), 3000), random_state=0)
            fig = px.scatter(sample, x=feat, y=x2, color="label", color_discrete_map=cmap,
                             category_orders={"label": order}, opacity=0.55,
                             hover_data={"video_id": True, "time_s": True},
                             title=f"{feat} vs {x2} (muestra de {len(sample):,} frames)")
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color=SURFACE)))
            show(fig, 450)

        with st.expander("Estadísticas descriptivas por clase"):
            st.dataframe(d.groupby("label")[FEATURES].agg(["mean", "std"]).reindex(order).round(3),
                         width="stretch")

        # --- lo que ve el modelo CNN: vectores de la ResNet18 reducidos a 2D
        with st.expander("Lo que ve el modelo CNN (ResNet18)"):
            pca = load_cnn_pca()
            dp = (d[["video_id", "time_s", "label"]].merge(pca.drop(columns="label"), on=["video_id", "time_s"])
                  if not pca.empty else pca)
            if dp.empty:
                st.caption("Falta `data/processed/cnn_pca.csv` (se genera con el notebook 07).")
            else:
                st.markdown("El modelo CNN no usa las 5 propiedades: describe cada frame con **512 valores** "
                            "de una ResNet18 preentrenada. Aquí se resumen en 2 ejes (PCA).")
                por = st.radio("Colorear por", ["Clase", "Video"], horizontal=True)
                # colores libres de la paleta validada; se asignan sobre todos los videos para que
                # un video no cambie de color al filtrar
                vmap = dict(zip(videos, ["#eda100", "#e87ba4", "#008300"]))
                col, cm_ = ("label", cmap) if por == "Clase" else ("video_id", vmap)
                sample = dp.sample(min(len(dp), 3000), random_state=0)
                fig = px.scatter(sample, x="pc1", y="pc2", color=col, color_discrete_map=cm_,
                                 category_orders={"label": order, "video_id": videos}, opacity=0.55,
                                 hover_data={"video_id": True, "time_s": True, "label": True},
                                 title=f"Frames en el espacio de la ResNet18, por {por.lower()} "
                                       f"(muestra de {len(sample):,})")
                fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
                fig.update_xaxes(title="Componente 1")
                fig.update_yaxes(title="Componente 2")
                show(fig, 450)

                # silhouette: 1 = grupos separados, 0 o negativo = grupos mezclados
                k = st.columns(2)
                for c, (nombre, g) in zip(k, [("clase (evento)", "label"), ("video (partido)", "video_id")]):
                    val = (silhouette_score(sample[["pc1", "pc2"]], sample[g])
                           if sample[g].nunique() > 1 else float("nan"))
                    c.metric(f"Separación por {nombre}", f"{val:.2f}", "silhouette, de -1 a 1",
                             delta_color="off")
                st.caption("Los frames se agrupan por **video**, no por **evento**: la red reconoce el "
                           "partido (estadio, luz, césped) mejor que la jugada. Por eso el modelo CNN "
                           "apenas supera al Dummy.")

        # --- galería
        st.subheader("Galería de frames")
        disp = d[d.existe]
        if not disp.empty:
            g1, g2 = st.columns([3, 1])
            gcls = g1.selectbox("Clase a mostrar", order, format_func=lambda c: CLASS_LABELS[c])
            n = g2.slider("Frames", 4, 16, 8, step=4)
            if st.button("🔀 Otros frames"):
                st.session_state["seed"] = st.session_state.get("seed", 0) + 1
            pick = disp[disp.label == gcls]
            pick = pick.sample(min(n, len(pick)), random_state=st.session_state.get("seed", 0))
            cols = st.columns(4)
            for i, r in enumerate(pick.itertuples(index=False)):
                cols[i % 4].image(str(ROOT / r.path), width="stretch",
                                  caption=f"{r.video_id} · t={r.time_s:.1f}s · verde={r.green_ratio:.2f}")
        else:
            st.caption("Los .jpg no están en el repo (se generan con el notebook 01). "
                       "Se muestran las galerías guardadas del EDA.")
            figs = {"Plano general": "06_plano_general.png", "Primer plano / repetición": "07_primer_plano.png",
                    "play": "09_galeria_play.png", "challenge": "09_galeria_challenge.png",
                    "throwin": "09_galeria_throwin.png"}
            figs = {k: v for k, v in figs.items() if (FIG_DIR / v).exists()}
            if figs:
                g = st.radio("Galería", list(figs), horizontal=True)
                st.image(str(FIG_DIR / figs[g]), width="stretch")

# ============================================================ CLASIFICAR
with tab_clf:
    st.subheader("Clasificar un frame nuevo")
    if not models:
        st.error("No hay modelos en `models/`.")
    else:
        keys = list(models)
        left, right = st.columns([2, 3])
        with left:
            up = st.file_uploader("Sube un frame del partido", type=["jpg", "jpeg", "png", "bmp", "webp"])
            modo = st.radio("Modelos", ["Todos", "Elegir"], horizontal=True)
            if modo == "Elegir":
                keys = st.multiselect("Modelos a usar", keys, default=keys[-1:], format_func=model_name)
            if up is not None:
                st.image(up, width="stretch", caption=up.name)

        with right:
            if up is None:
                st.info("Sube una imagen (por ejemplo, una captura de un partido) para ver qué "
                        "evento predice cada modelo.")
            elif not keys:
                st.warning("Elige al menos un modelo.")
            else:
                try:
                    img = decode_image(up.getvalue())
                except ValueError:
                    st.error("No se pudo leer la imagen. Prueba con otro archivo.")
                    st.stop()
                with st.spinner("Clasificando…"):
                    res = predict_all(img, keys, models)

                if "error" in res and res.error.notna().any():
                    for r in res[res.error.notna()].itertuples():
                        st.warning(f"{r.modelo}: no se pudo predecir ({r.error})")
                    res = res[res.error.isna()]

                if not res.empty:
                    f1 = (metrics.groupby("model").f1_macro.mean() if not metrics.empty
                          else pd.Series(dtype=float))
                    votos = res.pred.value_counts()
                    if len(res) > 1:
                        ganador = votos.index[0]
                        st.markdown(f"#### Voto de los modelos: **{CLASS_LABELS[ganador]}** "
                                    f"({votos.iloc[0]} de {len(res)})")
                    cards = st.columns(min(len(res), 3))
                    for i, r in enumerate(res.itertuples(index=False)):
                        p = getattr(r, f"proba_{r.pred}")
                        extra = f" · F1 macro en prueba {f1[r.model]:.2f}" if r.model in f1 else ""
                        cards[i % len(cards)].metric(r.modelo, CLASS_LABELS[r.pred],
                                                     f"confianza {p:.0%}{extra}", delta_color="off", delta_arrow="off")

                    long = res.melt(id_vars=["modelo"], value_vars=[f"proba_{c}" for c in CLASSES],
                                    var_name="clase", value_name="probabilidad")
                    long["clase"] = long.clase.str.replace("proba_", "", regex=False)
                    fig = px.bar(long, y="modelo", x="probabilidad", color="clase", orientation="h",
                                 color_discrete_map=CLASS_COLORS, category_orders={"clase": CLASSES},
                                 title="Probabilidad que asigna cada modelo a cada clase")
                    fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                                      hovertemplate="%{y} · %{fullData.name}: %{x:.1%}<extra></extra>")
                    fig.update_xaxes(tickformat=".0%", range=[0, 1], title=None)
                    fig.update_yaxes(title=None, autorange="reversed")
                    show(fig, 140 + 55 * len(res))

                    tabla = res.set_index("modelo")[[f"proba_{c}" for c in CLASSES]]
                    tabla.columns = CLASSES
                    st.dataframe(tabla.style.format("{:.1%}"), width="stretch")
                    st.caption("Los modelos se entrenaron con frames de transmisiones de la Bundesliga. "
                               "Imágenes de otro tipo (otro deporte, fotos) dan predicciones sin sentido.")

# ============================================================ RENDIMIENTO
with tab_perf:
    st.subheader("Rendimiento en los datos de prueba")
    st.markdown("Cada modelo se evaluó **dejando un video fuera** (LeaveOneGroupOut): se entrena "
                "con dos videos y se prueba en el tercero, así el modelo nunca ve el partido con el que se evalúa.")
    mostrar = st.toggle("Mostrar gráficas de rendimiento", value=True)

    if metrics.empty:
        st.info("Todavía no hay archivos `results/metrics_*.csv`.")
    elif mostrar:
        mkeys = sort_models(metrics.model.unique())
        sel = st.multiselect("Modelos a comparar", mkeys, default=mkeys, format_func=model_name)
        m = metrics[metrics.model.isin(sel)]
        p = preds[preds.model.isin(sel)] if not preds.empty else preds
        mcmap = model_color_map(sel)
        morder = [model_name(k) for k in sort_models(sel)]

        if m.empty:
            st.info("Elige al menos un modelo.")
        else:
            res = (m.groupby("modelo")[["accuracy", "f1_macro", "train_time_s", "pred_time_ms"]]
                   .mean().reindex(morder))
            best = res.f1_macro.idxmax()
            acc_best = res.accuracy.idxmax()
            k = st.columns(3)
            k[0].metric("Mejor F1 macro", best, f"{res.f1_macro.max():.3f}", delta_color="off", delta_arrow="off")
            k[1].metric("Mejor accuracy", acc_best, f"{res.accuracy.max():.3f}", delta_color="off", delta_arrow="off")
            k[2].metric("Videos de prueba (folds)", m.fold.nunique())
            if acc_best != best:
                st.warning(f"**El accuracy engaña:** *{acc_best}* gana en accuracy pero no en F1 macro. "
                           "Con tantas jugadas *play*, acertar casi solo esa clase da un accuracy alto.")

            # --- accuracy vs F1 macro, con variación entre videos
            a, b = st.columns(2)
            with a:
                agg = (m.melt(id_vars=["modelo"], value_vars=["accuracy", "f1_macro"],
                              var_name="métrica", value_name="valor")
                       .groupby(["métrica", "modelo"]).valor.agg(["mean", "std"]).reset_index())
                fig = px.bar(agg, x="métrica", y="mean", color="modelo", barmode="group",
                             error_y="std", color_discrete_map=mcmap,
                             category_orders={"modelo": morder},
                             title="Accuracy vs. F1 macro (media ± desv. entre videos)")
                fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                                  hovertemplate="%{fullData.name}: %{y:.3f}<extra></extra>")
                fig.update_yaxes(range=[0, 1], title=None)
                fig.update_xaxes(title=None)
                show(fig, 420)
            with b:
                f1c = [f"f1_{c}" for c in CLASSES if f"f1_{c}" in m]
                pc = (m.melt(id_vars=["modelo"], value_vars=f1c, var_name="clase", value_name="F1")
                      .groupby(["clase", "modelo"]).F1.mean().reset_index())
                pc["clase"] = pc.clase.str.replace("f1_", "", regex=False)
                fig = px.bar(pc, x="clase", y="F1", color="modelo", barmode="group",
                             color_discrete_map=mcmap, category_orders={"modelo": morder, "clase": CLASSES},
                             title="F1 por clase")
                fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                                  hovertemplate="%{x} · %{fullData.name}: %{y:.3f}<extra></extra>")
                fig.update_yaxes(range=[0, 1], title=None)
                fig.update_xaxes(title=None)
                show(fig, 420)

            # --- variación por video
            fig = px.strip(m, x="f1_macro", y="modelo", color="modelo", color_discrete_map=mcmap,
                           category_orders={"modelo": morder}, hover_data={"fold": True},
                           title="F1 macro en cada video de prueba")
            fig.update_traces(marker=dict(size=13, line=dict(width=2, color=SURFACE)), showlegend=False)
            fig.update_xaxes(title="F1 macro")
            fig.update_yaxes(title=None)
            show(fig, 120 + 50 * len(sel))

            # --- tiempos vs desempeño (dos gráficas, cada una con su escala)
            a, b = st.columns(2)
            r2 = res.reset_index()
            for col, t, lab in [(a, "train_time_s", "Tiempo de entrenamiento (s, escala log)"),
                                (b, "pred_time_ms", "Tiempo de predicción por frame (ms, escala log)")]:
                with col:
                    fig = px.scatter(r2, x=t, y="f1_macro", color="modelo", text="modelo",
                                     color_discrete_map=mcmap, log_x=True,
                                     category_orders={"modelo": morder},
                                     title="¿Vale la pena el costo? " + lab.split(" (")[0])
                    fig.update_traces(marker=dict(size=16, line=dict(width=2, color=SURFACE)),
                                      textposition="top center", showlegend=False, cliponaxis=False,
                                      hovertemplate="%{text}<br>%{x:.4g}<br>F1 macro %{y:.3f}<extra></extra>")
                    lo, hi = np.log10(r2[t].clip(lower=1e-6)).agg(["min", "max"])
                    fig.update_xaxes(title=lab, range=[lo - 0.6, hi + 0.6])
                    fig.update_yaxes(title="F1 macro")
                    show(fig)

            # --- matriz de confusión
            if not p.empty:
                st.markdown("#### ¿En qué se equivoca cada modelo?")
                c1, c2 = st.columns([1, 1])
                pkeys = sort_models(p.model.unique())
                best_key = next((k for k in pkeys if model_name(k) == best), pkeys[0])
                mk = c1.selectbox("Modelo", pkeys, format_func=model_name, index=pkeys.index(best_key))
                folds = ["Todos"] + sorted(p.video_id.unique())
                fsel = c2.selectbox("Video de prueba", folds)
                pm = p[p.model == mk]
                if fsel != "Todos":
                    pm = pm[pm.video_id == fsel]
                cm = pd.crosstab(pm.label, pm.pred).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
                pct = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
                fig = go.Figure(go.Heatmap(
                    z=pct.values, x=CLASSES, y=CLASSES, colorscale=[[0, "#262624"], [1, "#2f7fd8"]], textfont=dict(color="#f0efec", size=14), zmin=0, zmax=1,
                    text=[[f"{v:.0%}<br>({n:,})" for v, n in zip(rp, rn)]
                          for rp, rn in zip(pct.fillna(0).values, cm.values)],
                    texttemplate="%{text}", colorbar=dict(title="% de la fila", tickformat=".0%"),
                    hovertemplate="real %{y} → predicho %{x}<br>%{z:.1%} de la clase real<extra></extra>"))
                fig.update_layout(title=f"Matriz de confusión · {model_name(mk)}")
                fig.update_xaxes(title="Predicho", side="bottom")
                fig.update_yaxes(title="Real", autorange="reversed")
                c1, c2 = st.columns([3, 2])
                with c1:
                    show(fig, 440)
                with c2:
                    dist = (p.groupby(["modelo", "pred"]).size().rename("n").reset_index())
                    dist["%"] = dist.n / dist.groupby("modelo").n.transform("sum") * 100
                    real = p[p.model == p.model.iloc[0]].label.value_counts(normalize=True) * 100
                    dist = pd.concat([dist, pd.DataFrame({"modelo": "Real (etiquetas)", "pred": real.index,
                                                          "%": real.values})])
                    fig = px.bar(dist, y="modelo", x="%", color="pred", orientation="h",
                                 color_discrete_map=CLASS_COLORS,
                                 category_orders={"pred": CLASSES, "modelo": ["Real (etiquetas)"] + morder},
                                 title="Qué clases predice cada modelo")
                    fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                                      hovertemplate="%{y} · %{fullData.name}: %{x:.1f}%<extra></extra>")
                    fig.update_xaxes(range=[0, 100], title="% de predicciones")
                    fig.update_yaxes(title=None)
                    show(fig, 440)
                st.caption("Las filas de la matriz suman 100%: la diagonal es el recall de cada clase. "
                           "Si casi todo cae en la columna *play*, el modelo no está aprendiendo a separar los eventos.")

            with st.expander("Tabla de métricas por video"):
                cols = ["modelo", "fold", "accuracy", "f1_macro"] + [c for c in m if c.startswith("f1_") and c != "f1_macro"] \
                    + ["train_time_s", "pred_time_ms"]
                st.dataframe(m[cols].round(4), width="stretch", hide_index=True)
    else:
        st.caption("Gráficas ocultas. Activa el interruptor para verlas.")
