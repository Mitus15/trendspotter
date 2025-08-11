import os
import json
import io
import zipfile
import streamlit as st
from gemini_engine import GeminiAnalysisEngine


st.set_page_config(page_title="Gemini-Powered TrendSpotter Hub", layout="wide")
st.title("Gemini-Powered TrendSpotter Hub")


def _get_api_key() -> str | None:
    # Prefer Streamlit secrets; fallback to environment variable
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GEMINI_API_KEY")
    return key


api_key = _get_api_key()
if not api_key:
    st.error("Please set GEMINI_API_KEY in .streamlit/secrets.toml or environment.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_engine() -> GeminiAnalysisEngine:
    return GeminiAnalysisEngine(api_key=api_key)


engine = get_engine()

with st.sidebar:
    st.header("Configuration")
    steps = st.slider("Forecast horizon (months)", min_value=6, max_value=36, value=12, step=1)
    show_raw_outputs = st.toggle("Show raw model outputs", value=False)

class_map = engine.get_all_classes()
primary_classes = sorted(class_map.keys())

import numpy as np
import pandas as pd
import altair as alt
try:
    from vl_convert import vegalite_to_png  # type: ignore
except Exception:
    vegalite_to_png = None  # type: ignore
import re

def compute_kpis(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    if s.empty:
        return {}
    total = float(s.sum())
    last12 = float(s.tail(12).sum()) if len(s) >= 12 else float(s.sum())
    prev12 = float(s.tail(24).head(12).sum()) if len(s) >= 24 else np.nan
    growth_12m = (last12 / prev12 - 1.0) if prev12 and prev12 > 0 else np.nan
    pct_change = s.pct_change().dropna()
    vol = float(pct_change.std()) if not pct_change.empty else np.nan
    tail = s.tail(12)
    if len(tail) >= 3:
        x = np.arange(len(tail))
        slope = float(np.polyfit(x, tail.values, 1)[0])
    else:
        slope = np.nan
    return {"total_count": total, "last_12m": last12, "growth_12m": growth_12m, "volatility": vol, "momentum_slope": slope}

def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    base = df.replace(0, np.nan).iloc[0]
    return (df.divide(base).fillna(0.0) * 100.0).clip(upper=10000)

def chart_to_png_bytes(chart: alt.Chart) -> bytes | None:
    # Use vl-convert-python when available for reliability
    try:
        if vegalite_to_png is not None:
            spec = chart.to_dict()
            return vegalite_to_png(spec)
    except Exception:
        pass
    try:
        buf = io.BytesIO()
        chart.save(buf, format="png", method="vl-convert")
        return buf.getvalue()
    except Exception:
        try:
            buf = io.BytesIO()
            chart.save(buf, format="png")
            return buf.getvalue()
        except Exception:
            return None

def sanitize_filename(name: str) -> str:
    s = str(name)
    s = s.replace("—", "-").replace("/", "-").replace("|", "-")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.\-]", "", s)
    return s[:120] if len(s) > 120 else s

tab_explore, tab_compare, tab_report, tab_transparency = st.tabs(["Explore", "Compare", "Report", "Forecasting Analysis"])

with tab_explore:
    st.subheader("Explore a single category")
    if not primary_classes:
        st.info("Taxonomy not loaded yet. Try refreshing.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            p = st.selectbox("Primary", options=primary_classes, key="ex_p")
        with c2:
            s_opts = class_map.get(p, []) if p else []
            s = st.selectbox("Secondary", options=s_opts, key="ex_s")
        if p and s:
            ts = engine.get_monthly_series_by_class(p, s)
            if ts is None or ts.empty:
                st.warning("No data for selection.")
            else:
                # Plot single series using Altair explicitly
                ex_df = ts.reset_index()
                ex_df.columns = ["date", "value"]
                chart = (
                    alt.Chart(ex_df)
                    .mark_line()
                    .encode(x=alt.X("date:T", title="Month"), y=alt.Y("value:Q", title="Count"))
                    .properties(height=220)
                )
                st.altair_chart(chart, use_container_width=True)
                kpis = compute_kpis(ts)
                st.metric("Last 12m growth", None if np.isnan(kpis.get("growth_12m", np.nan)) else f"{kpis['growth_12m']*100:.1f}%")
                st.write(kpis)

                if st.button("Run ARIMA/GARCH diagnostics", key="ex_diag"):
                    with st.spinner("Fitting models and computing diagnostics..."):
                        label = f"{p} — {s}"
                        arima_res = engine.run_full_arima_analysis_from_series(ts, label, steps=steps)
                        garch_res = engine.run_full_garch_analysis_from_series(ts, label, steps=steps)

                        st.subheader("ARIMA diagnostics")
                        if "error" in arima_res:
                            st.warning(arima_res["error"])
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                st.caption("Candidate models (sorted by AIC)")
                                st.dataframe(pd.DataFrame(arima_res.get("candidate_models", [])))
                                st.caption("Stationarity (ADF)")
                                st.json(arima_res.get("stationarity", {}))
                            with c2:
                                st.caption("Residual diagnostics")
                                st.json(arima_res.get("residual_diagnostics", {}))
                                st.caption("Data coverage")
                                st.json(arima_res.get("data_coverage", {}))
                            with st.expander("ARIMA summary table"):
                                st.text(arima_res.get("summary_table", ""))
                            with st.expander("ARIMA forecast table"):
                                st.dataframe(pd.DataFrame(arima_res.get("forecast", [])))

                        st.subheader("GARCH diagnostics")
                        if "error" in garch_res:
                            st.warning(garch_res["error"])
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                st.caption("Params")
                                st.json(garch_res.get("model_params", {}))
                                st.caption("Stationarity (ADF on returns)")
                                st.json(garch_res.get("stationarity", {}))
                            with c2:
                                st.caption("Residual diagnostics")
                                st.json(garch_res.get("residual_diagnostics", {}))
                            with st.expander("GARCH summary table"):
                                st.text(garch_res.get("summary_table", ""))
                            with st.expander("Volatility forecast table"):
                                st.dataframe(pd.DataFrame(garch_res.get("volatility_forecast", [])))

with tab_compare:
    st.subheader("Compare up to three categories")
    selections = []
    cols = st.columns(3)
    for i, col in enumerate(cols, start=1):
        with col:
            p = st.selectbox(f"Primary {i}", options=[""] + primary_classes, key=f"cmp_p_{i}")
            s = st.selectbox(f"Secondary {i}", options=(class_map.get(p, []) if p else []), key=f"cmp_s_{i}")
            if p and s:
                selections.append({"primary": p, "secondary": s})
    normalize = st.checkbox("Normalize to index 100 (first available month)", value=True)
    if selections:
        aligned = []
        labels = []
        for sel in selections:
            # Avoid ':' in labels (Altair treats 'field:type'); also trim whitespace
            label = f"{sel['primary'].strip()} — {sel['secondary'].strip()}"
            ts = engine.get_monthly_series_by_class(sel['primary'], sel['secondary'])
            if ts is not None and not ts.empty:
                aligned.append(ts.rename(label))
                labels.append(label)
        if aligned:
            df = pd.concat(aligned, axis=1).asfreq("MS").fillna(0.0)
            # Extra safety: sanitize column names
            df.columns = [str(c).replace(":", " — ").replace("|", "/").strip() for c in df.columns]
            plot_df = normalize_index(df) if normalize else df
            # Long format for robust Altair charting
            lf = plot_df.reset_index().melt(id_vars=[plot_df.index.name or "index"], var_name="series", value_name="value")
            lf.columns = ["date", "series", "value"]
            comp_chart = (
                alt.Chart(lf)
                .mark_line()
                .encode(x=alt.X("date:T", title="Month"), y=alt.Y("value:Q", title="Index" if normalize else "Count"), color=alt.Color("series:N", title="Series"))
                .properties(height=260)
            )
            st.altair_chart(comp_chart, use_container_width=True)
            rows = []
            for label in labels:
                rows.append({"name": label, **compute_kpis(df[label])})
            st.dataframe(pd.DataFrame(rows).set_index("name"))
    else:
        st.info("Add at least one category to compare.")

with tab_report:
    st.subheader("Business-focused report")
    c1, c2, c3 = st.columns(3)
    with c1:
        region = st.text_input("Region/Market", value="Kamloops, BC")
    with c2:
        target_year = st.number_input("Target year", value=2026, min_value=2024, max_value=2035, step=1)
    with c3:
        risk = st.slider("Risk appetite", min_value=0, max_value=10, value=5)
    budget = st.selectbox("Budget posture", options=["lean", "balanced", "aggressive"], index=1)

    user_query = st.text_area(
        "Your question",
        value="Compare the future potential of selected categories. Which is the better investment and why?",
        height=120,
    )

    st.markdown("---")
    st.caption("Scope")
    scope_col1, scope_col2, scope_col3 = st.columns([1, 1, 1])
    with scope_col1:
        analyze_all = st.checkbox("Analyze all categories (auto)", value=False)
    with scope_col2:
        top_k = st.slider("Top K categories", min_value=5, max_value=30, value=12, step=1, disabled=not analyze_all)
    with scope_col3:
        rank_metric = st.selectbox("Rank by", options=["last_12m", "total_count", "momentum_slope"], index=0, disabled=not analyze_all)

    wcol1, wcol2, wcol3 = st.columns([1, 1, 1])
    with wcol1:
        weight_pop = st.slider("Weight: forecast popularity", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    with wcol2:
        weight_vol = st.slider("Weight: (1 - volatility)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    with wcol3:
        st.caption("Weights should sum to 1.0")

    run = st.button("Generate Business Report", type="primary")
    if run:
        selections = []
        if analyze_all:
            ranked = []
            for p_key, secs in class_map.items():
                for s_key in secs:
                    ts = engine.get_monthly_series_by_class(p_key, s_key)
                    if ts is None or ts.empty:
                        continue
                    k = compute_kpis(ts)
                    score = k.get(rank_metric)
                    try:
                        score_val = float(score) if score is not None and not np.isnan(score) else -1e12
                    except Exception:
                        score_val = -1e12
                    ranked.append({"primary": p_key, "secondary": s_key, "score": score_val, "kpis": k})
            ranked.sort(key=lambda x: x["score"], reverse=True)
            selections = [{"primary": r["primary"], "secondary": r["secondary"]} for r in ranked[:top_k]]
        else:
            for i in range(1, 4):
                p = st.session_state.get(f"cmp_p_{i}")
                s = st.session_state.get(f"cmp_s_{i}")
                if p and s:
                    selections.append({"primary": p, "secondary": s})

        with st.spinner("Running analyses and composing report..."):
            all_results = []
            for sel in selections:
                label = f"{sel['primary'].strip()} — {sel['secondary'].strip()}"
                ts = engine.get_monthly_series_by_class(sel['primary'], sel['secondary'])
                if ts is None or ts.empty:
                    continue
                all_results.append(engine.run_full_arima_analysis_from_series(ts, label, steps=steps))
                all_results.append(engine.run_full_garch_analysis_from_series(ts, label, steps=steps))

            # Aggregate ARIMA/GARCH per label
            label_to = {}
            for res in all_results:
                if "tag" not in res:
                    continue
                label = res["tag"]
                bucket = label_to.setdefault(label, {})
                if res.get("model_type") == "ARIMA":
                    bucket["arima"] = res
                elif res.get("model_type") == "GARCH":
                    bucket["garch"] = res

            # Compute forecast popularity (sum of ARIMA mean forecast) and volatility (avg GARCH volatility)
            rows = []
            for label, model_dict in label_to.items():
                pop = None
                vol = None
                if "arima" in model_dict:
                    f = model_dict["arima"].get("forecast", [])
                    try:
                        pop = float(sum([item.get("mean", 0.0) or 0.0 for item in f]))
                    except Exception:
                        pop = None
                if "garch" in model_dict:
                    vf = model_dict["garch"].get("volatility_forecast", [])
                    try:
                        vals = [item.get("volatility", None) for item in vf]
                        vals = [float(x) for x in vals if x is not None]
                        vol = float(sum(vals) / len(vals)) if vals else None
                    except Exception:
                        vol = None
                rows.append({"label": label, "forecast_popularity": pop, "avg_volatility": vol})

            if rows:
                df_rank = pd.DataFrame(rows).dropna(subset=["forecast_popularity", "avg_volatility"], how="all")
                # Normalize metrics
                eps = 1e-9
                if df_rank["forecast_popularity"].notna().any():
                    pmin, pmax = df_rank["forecast_popularity"].min(), df_rank["forecast_popularity"].max()
                    df_rank["pop_nrm"] = (df_rank["forecast_popularity"] - pmin) / (max(eps, pmax - pmin))
                else:
                    df_rank["pop_nrm"] = 0.0
                if df_rank["avg_volatility"].notna().any():
                    vmin, vmax = df_rank["avg_volatility"].min(), df_rank["avg_volatility"].max()
                    # lower volatility better -> invert
                    df_rank["vol_nrm"] = 1.0 - (df_rank["avg_volatility"] - vmin) / (max(eps, vmax - vmin))
                else:
                    df_rank["vol_nrm"] = 0.0
                # Composite index
                wsum = max(eps, (weight_pop + weight_vol))
                df_rank["composite_index"] = (weight_pop * df_rank["pop_nrm"] + weight_vol * df_rank["vol_nrm"]) / wsum
                df_rank = df_rank.sort_values("composite_index", ascending=False)

                st.subheader("Forecast-based ranking")
                st.dataframe(df_rank.set_index("label"))
                # Chart top 15
                top_chart = df_rank.head(15).melt(id_vars=["label"], value_vars=["pop_nrm", "vol_nrm", "composite_index"], var_name="metric", value_name="score")
                bar = (
                    alt.Chart(top_chart)
                    .mark_bar()
                    .encode(y=alt.Y("label:N", sort="-x", title="Category"), x=alt.X("score:Q", title="Score"), color=alt.Color("metric:N", title="Metric"))
                    .properties(height=400)
                )
                st.altair_chart(bar, use_container_width=True)

                # Identify best main category (exclude secondary 'other')
                best_main = None
                if not df_rank.empty:
                    non_other = df_rank[~df_rank["label"].str.endswith(" — other")]
                    if not non_other.empty:
                        best_main = non_other.iloc[0]["label"]

                # Deep dive: 'Other' inner tags – build top 5 by composite index
                st.subheader("'Other' category deep dive")
                # Collect candidate tags under 'Other'
                other_df = engine.df.copy()
                if "secondary_class" in other_df.columns:
                    other_df = other_df[other_df["secondary_class"].astype(str) == "other"]
                else:
                    other_df = other_df.iloc[0:0]
                inner_candidates = []
                # Preselect top by frequency to bound compute
                top_inner = (
                    other_df["canonical_tag" if "canonical_tag" in other_df.columns else "tag"].value_counts().head(50).index.tolist()
                )
                for inner in top_inner:
                    # Build series for this inner tag
                    col = "canonical_tag" if "canonical_tag" in engine.df.columns else "tag"
                    df_i = engine.df[engine.df[col].astype(str) == str(inner)]
                    if df_i.empty:
                        continue
                    ts_i = (
                        df_i.set_index("timestamp").sort_index().resample("MS").size().astype(float).asfreq("MS").fillna(0.0)
                    )
                    if len(ts_i) < 8:
                        continue
                    label_i = f"Other: {inner}"
                    arima_i = engine.run_full_arima_analysis_from_series(ts_i, label_i, steps=steps)
                    garch_i = engine.run_full_garch_analysis_from_series(ts_i, label_i, steps=steps)
                    # Compute metrics
                    pop_i = None
                    vol_i = None
                    if "forecast" in arima_i:
                        try:
                            pop_i = float(sum([x.get("mean", 0.0) or 0.0 for x in arima_i["forecast"]]))
                        except Exception:
                            pop_i = None
                    if "volatility_forecast" in garch_i:
                        try:
                            vv = [x.get("volatility", None) for x in garch_i["volatility_forecast"]]
                            vv = [float(x) for x in vv if x is not None]
                            vol_i = float(sum(vv) / len(vv)) if vv else None
                        except Exception:
                            vol_i = None
                    inner_candidates.append({
                        "label": label_i,
                        "pop": pop_i,
                        "vol": vol_i,
                        "arima": arima_i,
                        "garch": garch_i,
                        "series": ts_i,
                    })
                inner_df = pd.DataFrame([{ "label": c["label"], "forecast_popularity": c["pop"], "avg_volatility": c["vol"] } for c in inner_candidates])
                top_inner_rows = []
                if not inner_df.empty:
                    eps = 1e-9
                    pmin, pmax = inner_df["forecast_popularity"].min(), inner_df["forecast_popularity"].max()
                    inner_df["pop_nrm"] = (inner_df["forecast_popularity"] - pmin) / (max(eps, pmax - pmin))
                    vmin, vmax = inner_df["avg_volatility"].min(), inner_df["avg_volatility"].max()
                    inner_df["vol_nrm"] = 1.0 - (inner_df["avg_volatility"] - vmin) / (max(eps, vmax - vmin))
                    wsum = max(eps, (weight_pop + weight_vol))
                    inner_df["composite_index"] = (weight_pop * inner_df["pop_nrm"] + weight_vol * inner_df["vol_nrm"]) / wsum
                    inner_df = inner_df.sort_values("composite_index", ascending=False).head(5)
                    st.dataframe(inner_df.set_index("label"))
                    top_inner_labels = inner_df["label"].tolist()
                    # Visualize vibe index for best_main + top inner
                    viz_rows = []
                    for lbl in top_inner_labels:
                        r = inner_df[inner_df["label"] == lbl].iloc[0]
                        viz_rows.append({"label": lbl, "metric": "composite_index", "score": r["composite_index"]})
                    if best_main and best_main in df_rank["label"].values:
                        r = df_rank[df_rank["label"] == best_main].iloc[0]
                        viz_rows.append({"label": best_main, "metric": "composite_index", "score": r["composite_index"]})
                    if viz_rows:
                        vibe = pd.DataFrame(viz_rows)
                        chart_vibe = (
                            alt.Chart(vibe)
                            .mark_bar()
                            .encode(y=alt.Y("label:N", sort="-x", title="Category"), x=alt.X("score:Q", title="Vibe Index"), color=alt.Color("metric:N", legend=None))
                            .properties(height=300)
                        )
                        st.altair_chart(chart_vibe, use_container_width=True)

                    # Show ARIMA/GARCH forecast charts per selection (compact)
                    st.subheader("Forecast visuals (ARIMA mean and GARCH volatility)")
                    show_labels = [best_main] if best_main else []
                    show_labels += top_inner_labels
                    for lbl in show_labels:
                        st.markdown(f"**{lbl}**")
                        # Find data
                        c = next((c for c in inner_candidates if c["label"] == lbl), None)
                        if c is None and lbl in label_to:
                            # build from main pool
                            ar = label_to[lbl].get("arima")
                            gr = label_to[lbl].get("garch")
                        else:
                            ar = c.get("arima") if c else None
                            gr = c.get("garch") if c else None
                        if ar and "forecast" in ar:
                            fdf = pd.DataFrame(ar["forecast"])
                            fdf["date"] = pd.to_datetime(fdf["date"])  # ensure datetime
                            ar_chart = (
                                alt.Chart(fdf)
                                .mark_line(color="#1f77b4")
                                .encode(x=alt.X("date:T", title="Month"), y=alt.Y("mean:Q", title="ARIMA mean"))
                            )
                            band = (
                                alt.Chart(fdf)
                                .mark_area(opacity=0.2)
                                .encode(x=alt.X("date:T"), y="lower:Q", y2="upper:Q")
                            )
                            st.altair_chart(band + ar_chart, use_container_width=True)
                        if gr and "volatility_forecast" in gr:
                            vdf = pd.DataFrame(gr["volatility_forecast"])
                            vdf["date"] = pd.to_datetime(vdf["date"])  # ensure datetime
                            vol_chart = (
                                alt.Chart(vdf)
                                .mark_line(color="#d62728")
                                .encode(x=alt.X("date:T", title="Month"), y=alt.Y("volatility:Q", title="GARCH volatility"))
                            )
                            st.altair_chart(vol_chart, use_container_width=True)

            context = (
                f"Region: {region}. Target year: {target_year}. Risk appetite (0-10): {risk}. Budget posture: {budget}. "
                "Use MovieLens tag-count context (user-applied tags)."
            )
            prompt = (
                f"{user_query}\n\n"
                f"Business context: {context}\n\n"
                "If a large set of categories is provided, you should compare broadly, cluster related ones, and recommend an optimal portfolio mix. "
                "Favor categories with strong recent momentum, manageable volatility, and compelling strategic fit. Include ARIMA/GARCH diagnostics where available. "
                "Use forecasted popularity (ARIMA mean) and volatility (GARCH) to rank categories and justify the recommended mix."
            )
            if show_raw_outputs and all_results:
                with st.expander("Raw analysis outputs (JSON)"):
                    st.json(all_results)
            report = engine.generate_report(all_results, prompt)
            st.markdown(report)


with tab_transparency:
    st.subheader("Full Forecasting Analysis and Audit Bundle")
    st.caption("Run a batch analysis and export every intermediate artifact: series, forecasts, diagnostics, parameters, tests, and methodology notes.")

    # Controls
    analyze_all_t = st.checkbox("Analyze all categories (capped)", value=False)
    cap_t = st.slider("Max categories", min_value=3, max_value=50, value=10, step=1)
    steps_t = st.slider("Forecast horizon (months)", min_value=6, max_value=36, value=12, step=1, key="steps_t")
    wcol1_t, wcol2_t = st.columns([1, 1])
    with wcol1_t:
        weight_pop_other = st.slider("Weight: forecast popularity (Vibe Index)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    with wcol2_t:
        weight_vol_other = st.slider("Weight: (1 - volatility) (Vibe Index)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    ccol1, ccol2 = st.columns([1, 1])
    with ccol1:
        risk_penalty_gamma = st.slider("Risk penalty curvature (γ)", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                                       help=">1 increases punishment on higher volatility")
    with ccol2:
        reward_dampen = st.slider("Reward dampening by popularity", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                  help="How much high popularity reduces the volatility penalty")
    run_t = st.button("Run and Build Audit Bundle", type="primary")
    run_full_zip = st.button("Run Full Analysis and Download ZIP", type="secondary")

    if run_t or run_full_zip:
        selections_t = []
        if analyze_all_t:
            # Prioritize by frequency to keep compute bounded deterministically
            freq = engine.df.groupby(["primary_class", "secondary_class"]).size().reset_index(name="n").sort_values("n", ascending=False)
            for _, row in freq.head(cap_t).iterrows():
                selections_t.append({"primary": row["primary_class"], "secondary": row["secondary_class"]})
        else:
            for i in range(1, 4):
                p = st.session_state.get(f"cmp_p_{i}")
                s = st.session_state.get(f"cmp_s_{i}")
                if p and s:
                    selections_t.append({"primary": p, "secondary": s})

        # If no selections provided, fall back to top categories by frequency automatically
        if not selections_t:
            freq = engine.df.groupby(["primary_class", "secondary_class"]).size().reset_index(name="n").sort_values("n", ascending=False)
            for _, row in freq.head(cap_t).iterrows():
                selections_t.append({"primary": row["primary_class"], "secondary": row["secondary_class"]})

        with st.spinner("Running full diagnostics for forecasting analysis..."):
            bundle = {"meta": {"steps": steps_t}, "items": []}
            for sel in selections_t:
                label = f"{sel['primary'].strip()} — {sel['secondary'].strip()}"
                ts = engine.get_monthly_series_by_class(sel['primary'], sel['secondary'])
                if ts is None or ts.empty:
                    continue
                ar = engine.run_full_arima_analysis_from_series(ts, label, steps=steps_t)
                gr = engine.run_full_garch_analysis_from_series(ts, label, steps=steps_t)
                series_df = ts.reset_index()
                series_df.columns = ["date", "value"]
                item = {
                    "label": label,
                    "series": series_df.assign(date=series_df["date"].dt.date.astype(str)).to_dict(orient="records"),
                    "arima": ar,
                    "garch": gr,
                }
                bundle["items"].append(item)

            # Deep dive: 'Other' inner tags – compute top Vibe Index subtags and include in bundle
            other_df_t = engine.df.copy()
            if "secondary_class" in other_df_t.columns:
                other_df_t = other_df_t[other_df_t["secondary_class"].astype(str) == "other"]
            else:
                other_df_t = other_df_t.iloc[0:0]

            inner_candidates_t = []
            if not other_df_t.empty:
                col_t = "canonical_tag" if "canonical_tag" in other_df_t.columns else "tag"
                top_inner_t = other_df_t[col_t].value_counts().head(50).index.tolist()
                for inner in top_inner_t:
                    df_i_t = engine.df[engine.df[col_t].astype(str) == str(inner)]
                    if df_i_t.empty:
                        continue
                    ts_i_t = (
                        df_i_t.set_index("timestamp").sort_index().resample("MS").size().astype(float).asfreq("MS").fillna(0.0)
                    )
                    if len(ts_i_t) < 8:
                        continue
                    label_i_t = f"Other: {inner}"
                    arima_i_t = engine.run_full_arima_analysis_from_series(ts_i_t, label_i_t, steps=steps_t)
                    garch_i_t = engine.run_full_garch_analysis_from_series(ts_i_t, label_i_t, steps=steps_t)
                    pop_i_t = None
                    vol_i_t = None
                    if "forecast" in arima_i_t:
                        try:
                            pop_i_t = float(sum([x.get("mean", 0.0) or 0.0 for x in arima_i_t["forecast"]]))
                        except Exception:
                            pop_i_t = None
                    if "volatility_forecast" in garch_i_t:
                        try:
                            vv_t = [x.get("volatility", None) for x in garch_i_t["volatility_forecast"]]
                            vv_t = [float(x) for x in vv_t if x is not None]
                            vol_i_t = float(sum(vv_t) / len(vv_t)) if vv_t else None
                        except Exception:
                            vol_i_t = None
                    # capture small series for stats/plots
                    series_i_t = ts_i_t.reset_index()
                    series_i_t.columns = ["date", "value"]
                    inner_candidates_t.append({
                        "label": label_i_t,
                        "forecast_popularity": pop_i_t,
                        "avg_volatility": vol_i_t,
                        "arima": arima_i_t,
                        "garch": garch_i_t,
                        "series": series_i_t.assign(date=series_i_t["date"].dt.date.astype(str)).to_dict(orient="records"),
                    })

            inner_df_t = pd.DataFrame(inner_candidates_t)
            rankings_records = []
            if not inner_df_t.empty:
                eps = 1e-9
                pmin_t, pmax_t = inner_df_t["forecast_popularity"].min(), inner_df_t["forecast_popularity"].max()
                inner_df_t["pop_nrm"] = (inner_df_t["forecast_popularity"] - pmin_t) / (max(eps, pmax_t - pmin_t))
                vmin_t, vmax_t = inner_df_t["avg_volatility"].min(), inner_df_t["avg_volatility"].max()
                vol_raw = (inner_df_t["avg_volatility"] - vmin_t) / (max(eps, vmax_t - vmin_t))
                risk = vol_raw.pow(risk_penalty_gamma)
                eff_risk = risk * (1.0 - reward_dampen * inner_df_t["pop_nrm"])
                inner_df_t["vol_nrm"] = 1.0 - eff_risk
                wsum_t = max(eps, (weight_pop_other + weight_vol_other))
                inner_df_t["composite_index"] = (weight_pop_other * inner_df_t["pop_nrm"] + weight_vol_other * inner_df_t["vol_nrm"]) / wsum_t
                inner_df_t = inner_df_t.sort_values("composite_index", ascending=False)

                st.subheader("'Other' subtags by Vibe Index (Forecasting Analysis)")
                df_view_inner = inner_df_t[["label", "forecast_popularity", "avg_volatility", "pop_nrm", "vol_nrm", "composite_index"]].set_index("label").head(15)
                st.dataframe(df_view_inner)

                # Bar chart for top 15
                _viz_t = inner_df_t.head(15).melt(id_vars=["label"], value_vars=["composite_index"], var_name="metric", value_name="score")
                bar_t = (
                    alt.Chart(_viz_t)
                    .mark_bar()
                    .encode(y=alt.Y("label:N", sort="-x", title="Subtag"), x=alt.X("score:Q", title="Vibe Index"), color=alt.Color("metric:N", legend=None))
                    .properties(title="Top 'Other' subtags by Vibe Index", height=360)
                )
                st.altair_chart(bar_t, use_container_width=True)
                png_bar = chart_to_png_bytes(bar_t)
                if png_bar:
                    st.download_button("Download bar chart (PNG)", data=png_bar, file_name="other_subtags_vibe_bar.png", mime="image/png")

                # Prepare bundle records and CSV
                rankings_records = inner_df_t[["label", "forecast_popularity", "avg_volatility", "pop_nrm", "vol_nrm", "composite_index"]].to_dict(orient="records")
                csv_t = inner_df_t[["label", "forecast_popularity", "avg_volatility", "pop_nrm", "vol_nrm", "composite_index"]].to_csv(index=False)
                st.download_button("Download 'Other' subtag rankings (CSV)", data=csv_t, file_name="other_subtag_vibe_rankings.csv", mime="text/csv")

            # Compute vibe index across main items as well; consolidate and rank top X
            main_metrics_rows = []
            for it in bundle["items"]:
                pop = None
                vol = None
                lbl_m = str(it.get("label") or "").strip()
                # Exclude umbrella 'Other' from ranking
                if lbl_m.lower().endswith(" — other"):
                    continue
                try:
                    f = (it.get("arima", {}) or {}).get("forecast", [])
                    pop = float(sum([x.get("mean", 0.0) or 0.0 for x in f])) if f else None
                except Exception:
                    pop = None
                try:
                    vf = (it.get("garch", {}) or {}).get("volatility_forecast", [])
                    vv = [x.get("volatility", None) for x in vf]
                    vv = [float(x) for x in vv if x is not None]
                    vol = float(sum(vv) / len(vv)) if vv else None
                except Exception:
                    vol = None
                main_metrics_rows.append({"label": lbl_m, "source": "main", "forecast_popularity": pop, "avg_volatility": vol})

            rank_all = pd.DataFrame(main_metrics_rows)
            if not inner_df_t.empty:
                rank_all = pd.concat([rank_all, inner_df_t[["label", "forecast_popularity", "avg_volatility"]].assign(source="other")], ignore_index=True)
            else:
                rank_all = rank_all

            rank_all = rank_all.dropna(subset=["forecast_popularity", "avg_volatility"], how="any")
            vibe_records_all = []
            top_labels = []
            if not rank_all.empty:
                eps = 1e-9
                pmin_a, pmax_a = rank_all["forecast_popularity"].min(), rank_all["forecast_popularity"].max()
                vmin_a, vmax_a = rank_all["avg_volatility"].min(), rank_all["avg_volatility"].max()
                rank_all["pop_nrm"] = (rank_all["forecast_popularity"] - pmin_a) / (max(eps, pmax_a - pmin_a))
                vol_raw_a = (rank_all["avg_volatility"] - vmin_a) / (max(eps, vmax_a - vmin_a))
                risk_a = vol_raw_a.pow(risk_penalty_gamma)
                eff_risk_a = risk_a * (1.0 - reward_dampen * rank_all["pop_nrm"])
                rank_all["vol_nrm"] = 1.0 - eff_risk_a
                wsum_a = max(eps, (weight_pop_other + weight_vol_other))
                rank_all["composite_index"] = (weight_pop_other * rank_all["pop_nrm"] + weight_vol_other * rank_all["vol_nrm"]) / wsum_a
                rank_all_sorted = rank_all.sort_values("composite_index", ascending=False)
                top_labels = rank_all_sorted.head(cap_t)["label"].tolist()
                vibe_records_all = rank_all_sorted.to_dict(orient="records")

            # Build top selection diagnostics and attach to bundle
            top_selection = []
            for lbl in top_labels:
                # locate record source
                src = None
                it = next((x for x in bundle["items"] if x["label"] == lbl), None)
                inner = next((x for x in (inner_candidates_t or []) if x.get("label") == lbl), None)
                if it is not None:
                    src = "main"
                    series_records = it.get("series", [])
                    ar = it.get("arima", {}) or {}
                    gr = it.get("garch", {}) or {}
                elif inner is not None:
                    src = "other"
                    series_records = inner.get("series", [])
                    ar = inner.get("arima", {}) or {}
                    gr = inner.get("garch", {}) or {}
                else:
                    continue
                # descriptive statistics
                desc = {"mean": None, "median": None, "std": None}
                try:
                    sdf = pd.DataFrame(series_records)
                    vals = pd.to_numeric(sdf.get("value", pd.Series(dtype=float)), errors="coerce").dropna()
                    desc = {
                        "mean": float(vals.mean()) if len(vals) else None,
                        "median": float(vals.median()) if len(vals) else None,
                        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    }
                except Exception:
                    pass
                # ARIMA summary
                ar_summary = {
                    "best_order": ar.get("best_order"),
                    "aic_score": ar.get("aic_score"),
                }
                # GARCH summary
                params = gr.get("model_params", {}) or {}
                arch_test = gr.get("arch_effect_test", {}) or {}
                gr_summary = {
                    "alpha[1]": params.get("alpha[1]"),
                    "beta[1]": params.get("beta[1]"),
                    "arch_lm_pvalue": arch_test.get("lm_pvalue"),
                }
                top_selection.append({
                    "label": lbl,
                    "source": src,
                    "descriptive_statistics": desc,
                    "arima_summary": ar_summary,
                    "garch_summary": gr_summary,
                    "series": series_records,
                    "arima_forecast": ar.get("forecast", []),
                    "garch_volatility_forecast": gr.get("volatility_forecast", []),
                })

            # Attach to audit bundle
            bundle["other_inner_rankings"] = rankings_records
            bundle["vibe_rankings_all"] = vibe_records_all
            bundle["top_selection"] = top_selection

            text = json.dumps(bundle, ensure_ascii=False, indent=2)
            st.download_button("Download audit bundle (JSON)", data=text, file_name="audit_bundle.json", mime="application/json")

            # If full ZIP requested: generate charts for top-X by vibe and package everything
            if run_full_zip:
                # Build per-selection charts and collect
                png_buffers_all: list[tuple[str, bytes]] = []
                charts_meta: list[dict] = []
                for lbl in top_labels:
                    # locate artifacts
                    it = next((x for x in bundle["items"] if x["label"] == lbl), None)
                    inner = next((x for x in (inner_candidates_t or []) if x.get("label") == lbl), None)
                    if it is not None:
                        series_records = it.get("series", [])
                        ar = it.get("arima", {}) or {}
                        gr = it.get("garch", {}) or {}
                    elif inner is not None:
                        series_records = inner.get("series", [])
                        ar = inner.get("arima", {}) or {}
                        gr = inner.get("garch", {}) or {}
                    else:
                        continue
                    # Build charts
                    try:
                        hist_df = pd.DataFrame(series_records)
                        hist_df["date"] = pd.to_datetime(hist_df["date"])  # ensure datetime
                        hist_df["value"] = pd.to_numeric(hist_df["value"], errors="coerce").fillna(0.0)
                        hist_chart = (
                            alt.Chart(hist_df)
                            .mark_line(color="#1f77b4")
                            .encode(x=alt.X("date:T", title="Month"), y=alt.Y("value:Q", title="Count"))
                            .properties(height=220)
                        )
                        overlay = hist_chart
                        if ar and "forecast" in ar:
                            fdf = pd.DataFrame(ar["forecast"]).copy()
                            fdf["date"] = pd.to_datetime(fdf["date"])  # ensure datetime
                            fc_line = (
                                alt.Chart(fdf)
                                .mark_line(color="#ff7f0e")
                                .encode(x=alt.X("date:T"), y=alt.Y("mean:Q", title="Count"))
                            )
                            band = (
                                alt.Chart(fdf)
                                .mark_area(opacity=0.15, color="#ff7f0e")
                                .encode(x=alt.X("date:T"), y="lower:Q", y2="upper:Q")
                            )
                            overlay = hist_chart + band + fc_line
                        png_overlay = chart_to_png_bytes(overlay)
                        if png_overlay:
                            fname = f"charts/{sanitize_filename(lbl)}__arima.png"
                            png_buffers_all.append((fname, png_overlay))
                        # GARCH chart
                        vdf = pd.DataFrame(gr.get("volatility_forecast", []))
                        if not vdf.empty:
                            vdf["date"] = pd.to_datetime(vdf["date"])  # ensure datetime
                            vol_chart = (
                                alt.Chart(vdf)
                                .mark_line(color="#d62728")
                                .encode(x=alt.X("date:T", title="Month"), y=alt.Y("volatility:Q", title="GARCH volatility"))
                                .properties(height=180)
                            )
                            png_vol = chart_to_png_bytes(vol_chart)
                            if png_vol:
                                fname = f"charts/{sanitize_filename(lbl)}__garch_vol.png"
                                png_buffers_all.append((fname, png_vol))
                        charts_meta.append({"label": lbl})
                    except Exception:
                        continue

                # Build ZIP: include JSONs + charts (and 'Other' Top3 extras)
                zip_bytes = io.BytesIO()
                with zipfile.ZipFile(zip_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    # audit bundle
                    zf.writestr("audit/audit_bundle.json", text)
                    # vibe rankings and top selection as separate JSONs for convenience
                    zf.writestr("audit/vibe_rankings_all.json", json.dumps(vibe_records_all, ensure_ascii=False, indent=2))
                    zf.writestr("audit/top_selection.json", json.dumps(top_selection, ensure_ascii=False, indent=2))
                    # charts
                    for fname, data in png_buffers_all:
                        zf.writestr(fname, data)
                    # if we computed inner_df_t, include as CSV
                    try:
                        if not inner_df_t.empty:
                            zf.writestr("audit/other_inner_rankings.csv", inner_df_t[["label","forecast_popularity","avg_volatility","pop_nrm","vol_nrm","composite_index"]].to_csv(index=False))
                            # Compute 'Other' Top 3 and include JSON/CSV and charts
                            other_only = inner_df_t.sort_values("composite_index", ascending=False)
                            other_top3 = other_only.head(3)[["label","forecast_popularity","avg_volatility","pop_nrm","vol_nrm","composite_index"]]
                            zf.writestr("audit/other_top3.csv", other_top3.to_csv(index=False))
                            # Build top3 details JSON
                            top3_details = []
                            for lbl in other_top3["label"].tolist():
                                rec = next((x for x in (inner_candidates_t or []) if x.get("label") == lbl), None)
                                if not rec:
                                    continue
                                # Minimal stats
                                try:
                                    sdf = pd.DataFrame(rec.get("series", []))
                                    vals = pd.to_numeric(sdf.get("value", pd.Series(dtype=float)), errors="coerce").dropna()
                                    stats = {
                                        "mean": float(vals.mean()) if len(vals) else None,
                                        "median": float(vals.median()) if len(vals) else None,
                                        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                                    }
                                except Exception:
                                    stats = {"mean": None, "median": None, "std": None}
                                ar = rec.get("arima", {}) or {}
                                gr = rec.get("garch", {}) or {}
                                params = gr.get("model_params", {}) or {}
                                arch_test = gr.get("arch_effect_test", {}) or {}
                                top3_details.append({
                                    "label": lbl,
                                    "descriptive_statistics": stats,
                                    "arima_summary": {"best_order": ar.get("best_order"), "aic_score": ar.get("aic_score")},
                                    "garch_summary": {"alpha[1]": params.get("alpha[1]"), "beta[1]": params.get("beta[1]"), "arch_lm_pvalue": arch_test.get("lm_pvalue")},
                                    "series": rec.get("series", []),
                                    "arima_forecast": ar.get("forecast", []),
                                    "garch_volatility_forecast": gr.get("volatility_forecast", []),
                                })
                                # Charts for top3
                                try:
                                    # ARIMA overlay
                                    hist_df = pd.DataFrame(rec.get("series", []))
                                    hist_df["date"] = pd.to_datetime(hist_df["date"])  # ensure datetime
                                    hist_df["value"] = pd.to_numeric(hist_df["value"], errors="coerce").fillna(0.0)
                                    hist_chart = (
                                        alt.Chart(hist_df)
                                        .mark_line(color="#1f77b4")
                                        .encode(x=alt.X("date:T", title="Month"), y=alt.Y("value:Q", title="Count"))
                                        .properties(title=f"{lbl} — Historical and ARIMA forecast", height=220)
                                    )
                                    overlay = hist_chart
                                    if ar and "forecast" in ar:
                                        fdf = pd.DataFrame(ar["forecast"]).copy()
                                        fdf["date"] = pd.to_datetime(fdf["date"])  # ensure datetime
                                        fc_line = (
                                            alt.Chart(fdf)
                                            .mark_line(color="#ff7f0e")
                                            .encode(x=alt.X("date:T"), y=alt.Y("mean:Q", title="Count"))
                                        )
                                        band = (
                                            alt.Chart(fdf)
                                            .mark_area(opacity=0.15, color="#ff7f0e")
                                            .encode(x=alt.X("date:T"), y="lower:Q", y2="upper:Q")
                                        )
                                        overlay = hist_chart + band + fc_line
                                    png_ar = chart_to_png_bytes(overlay)
                                    if png_ar:
                                        zf.writestr(f"charts/other_top3/{sanitize_filename(lbl)}__arima.png", png_ar)
                                    # GARCH vol
                                    vdf = pd.DataFrame((gr or {}).get("volatility_forecast", []))
                                    if not vdf.empty:
                                        vdf["date"] = pd.to_datetime(vdf["date"])  # ensure datetime
                                        vol_chart = (
                                            alt.Chart(vdf)
                                            .mark_line(color="#d62728")
                                            .encode(x=alt.X("date:T", title="Month"), y=alt.Y("volatility:Q", title="GARCH volatility"))
                                            .properties(title=f"{lbl} — GARCH volatility forecast", height=180)
                                        )
                                        png_vol = chart_to_png_bytes(vol_chart)
                                        if png_vol:
                                            zf.writestr(f"charts/other_top3/{sanitize_filename(lbl)}__garch_vol.png", png_vol)
                                except Exception:
                                    pass
                            zf.writestr("audit/other_top3.json", json.dumps(top3_details, ensure_ascii=False, indent=2))
                    except Exception:
                        pass
                st.download_button("Download Full Analysis (ZIP)", data=zip_bytes.getvalue(), file_name="full_analysis_bundle.zip", mime="application/zip")

            # Interactive diagnostics explorer
            st.markdown("---")
            st.subheader("Diagnostics Explorer: Compare main genres and top 'Other' subtags")

            # Build options (top X by Vibe Index across main + other); fallback to all if ranking empty
            main_labels = [item["label"] for item in bundle["items"]]
            inner_label_to = {c["label"]: c for c in (inner_candidates_t or [])}
            if 'rank_all' in locals() and not rank_all.empty:
                all_options = rank_all_sorted["label"].tolist()
            else:
                all_options = main_labels + list(inner_label_to.keys())
            if not all_options:
                st.info("No selections available. Run the analysis above first.")
            else:
                default_opts = all_options[: min(cap_t, len(all_options))]
                selected = st.multiselect(
                    "Select up to X categories (defaults to top X by Vibe Index)",
                    options=all_options,
                    default=default_opts,
                    key="diag_select_transparency",
                )
                selected = selected[: cap_t]

                if selected:
                    # Descriptive statistics
                    stats_rows = []
                    for lbl in selected:
                        # locate series
                        series_records = None
                        if lbl in main_labels:
                            it = next((x for x in bundle["items"] if x["label"] == lbl), None)
                            if it:
                                series_records = it.get("series", [])
                        elif lbl in inner_label_to:
                            series_records = inner_label_to[lbl].get("series", [])
                        if not series_records:
                            continue
                        try:
                            sdf = pd.DataFrame(series_records)
                            sdf["value"] = pd.to_numeric(sdf["value"], errors="coerce").fillna(0.0)
                            stats_rows.append({
                                "label": lbl,
                                "mean": float(sdf["value"].mean()),
                                "median": float(sdf["value"].median()),
                                "std": float(sdf["value"].std(ddof=1) if len(sdf) > 1 else 0.0),
                            })
                        except Exception:
                            pass
                    if stats_rows:
                        st.caption("Descriptive statistics of historical series")
                        st.dataframe(pd.DataFrame(stats_rows).set_index("label"))

                    # ARIMA summary table
                    arima_rows = []
                    for lbl in selected:
                        if lbl in main_labels:
                            it = next((x for x in bundle["items"] if x["label"] == lbl), None)
                            ar = (it or {}).get("arima", {})
                        else:
                            ar = inner_label_to.get(lbl, {}).get("arima", {})
                        arima_rows.append({
                            "label": lbl,
                            "best_order": ar.get("best_order"),
                            "aic_score": ar.get("aic_score"),
                        })
                    st.caption("ARIMA model summary")
                    st.dataframe(pd.DataFrame(arima_rows).set_index("label"))

                    # GARCH summary table
                    garch_rows = []
                    for lbl in selected:
                        if lbl in main_labels:
                            it = next((x for x in bundle["items"] if x["label"] == lbl), None)
                            gr = (it or {}).get("garch", {})
                        else:
                            gr = inner_label_to.get(lbl, {}).get("garch", {})
                        params = gr.get("model_params", {}) or {}
                        alpha1 = params.get("alpha[1]")
                        beta1 = params.get("beta[1]")
                        lm_p = None
                        # Prefer arch_effect_test if available, else residual_diagnostics
                        arch_test = gr.get("arch_effect_test", {})
                        if arch_test:
                            lm_p = arch_test.get("lm_pvalue")
                        garch_rows.append({
                            "label": lbl,
                            "alpha[1]": alpha1,
                            "beta[1]": beta1,
                            "ARCH-LM p-value": lm_p,
                        })
                    st.caption("GARCH model summary")
                    st.dataframe(pd.DataFrame(garch_rows).set_index("label"))

                    # Visuals per selection
                    png_buffers = []
                    for lbl in selected:
                        st.markdown(f"**{lbl}**")
                        # Historical series and ARIMA forecast
                        if lbl in main_labels:
                            it = next((x for x in bundle["items"] if x["label"] == lbl), None)
                            series_records = (it or {}).get("series", [])
                            ar = (it or {}).get("arima", {})
                        else:
                            series_records = inner_label_to.get(lbl, {}).get("series", [])
                            ar = inner_label_to.get(lbl, {}).get("arima", {})

                        try:
                            hist_df = pd.DataFrame(series_records)
                            hist_df["date"] = pd.to_datetime(hist_df["date"])  # ensure datetime
                            hist_df["value"] = pd.to_numeric(hist_df["value"], errors="coerce").fillna(0.0)
                            hist_chart = (
                                alt.Chart(hist_df)
                                .mark_line(color="#1f77b4")
                                .encode(x=alt.X("date:T", title="Month"), y=alt.Y("value:Q", title="Count"))
                                .properties(height=220)
                            )
                            overlay = hist_chart
                            if ar and "forecast" in ar:
                                fdf = pd.DataFrame(ar["forecast"]).copy()
                                fdf["date"] = pd.to_datetime(fdf["date"])  # ensure datetime
                                fc_line = (
                                    alt.Chart(fdf)
                                    .mark_line(color="#ff7f0e")
                                    .encode(x=alt.X("date:T"), y=alt.Y("mean:Q", title="Count"))
                                )
                                band = (
                                    alt.Chart(fdf)
                                    .mark_area(opacity=0.15, color="#ff7f0e")
                                    .encode(x=alt.X("date:T"), y="lower:Q", y2="upper:Q")
                                )
                            overlay = (hist_chart + band + fc_line).properties(title=f"{lbl} — Historical and ARIMA forecast")
                            st.altair_chart(overlay, use_container_width=True)
                            png_overlay = chart_to_png_bytes(overlay)
                            if png_overlay:
                                fname = f"arima_{lbl.replace(' ', '_')}.png"
                                st.download_button(f"Download ARIMA chart (PNG) — {lbl}", data=png_overlay, file_name=fname, mime="image/png")
                                png_buffers.append((fname, png_overlay))
                        except Exception:
                            pass

                        # GARCH volatility forecast
                        if lbl in main_labels:
                            it = next((x for x in bundle["items"] if x["label"] == lbl), None)
                            gr = (it or {}).get("garch", {})
                        else:
                            gr = inner_label_to.get(lbl, {}).get("garch", {})
                        try:
                            vdf = pd.DataFrame(gr.get("volatility_forecast", []))
                            if not vdf.empty:
                                vdf["date"] = pd.to_datetime(vdf["date"])  # ensure datetime
                            vol_chart = (
                                alt.Chart(vdf)
                                .mark_line(color="#d62728")
                                .encode(x=alt.X("date:T", title="Month"), y=alt.Y("volatility:Q", title="GARCH volatility"))
                                .properties(title=f"{lbl} — GARCH volatility forecast", height=180)
                            )
                            st.altair_chart(vol_chart, use_container_width=True)
                            png_vol = chart_to_png_bytes(vol_chart)
                            if png_vol:
                                fname = f"garch_vol_{lbl.replace(' ', '_')}.png"
                                st.download_button(f"Download GARCH volatility (PNG) — {lbl}", data=png_vol, file_name=fname, mime="image/png")
                                png_buffers.append((fname, png_vol))
                        except Exception:
                            pass

                    # ZIP download of all generated PNGs for selected set
                    if png_buffers:
                        zip_bytes = io.BytesIO()
                        with zipfile.ZipFile(zip_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                            for fname, data in png_buffers:
                                zf.writestr(fname, data)
                        st.download_button("Download all charts (ZIP)", data=zip_bytes.getvalue(), file_name="charts_bundle.zip", mime="application/zip")

            st.markdown("---")
            st.caption("Preview of first item diagnostics")
            if bundle["items"]:
                first = bundle["items"][0]
                st.json({
                    "label": first["label"],
                    "arima_keys": list(first["arima"].keys()),
                    "garch_keys": list(first["garch"].keys()),
                })


    