from __future__ import annotations

import os
import io
import re
import json
import math
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import altair as alt
try:
    from vl_convert import vegalite_to_png  # type: ignore
except Exception:
    vegalite_to_png = None  # type: ignore

from gemini_engine import GeminiAnalysisEngine


def sanitize_filename(name: str) -> str:
    s = str(name)
    s = s.replace("—", "-").replace("/", "-").replace("|", "-")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.\-]", "", s)
    return s[:120] if len(s) > 120 else s


def chart_to_png_bytes(chart: alt.Chart) -> bytes | None:
    # First, try using vl-convert-python directly for better reliability
    try:
        if vegalite_to_png is not None:
            spec = chart.to_dict()
            return vegalite_to_png(spec)
    except Exception:
        pass
    # Fallback to Altair's save with vl-convert backend
    try:
        buf = io.BytesIO()
        chart.save(buf, format="png", method="vl-convert")
        return buf.getvalue()
    except Exception:
        # Final fallback to Altair default (may require node or selenium backends)
        try:
            buf = io.BytesIO()
            chart.save(buf, format="png")
            return buf.getvalue()
        except Exception:
            return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export full analysis bundle: JSON audits and PNG charts")
    p.add_argument("--steps", type=int, default=12, help="Forecast horizon (months)")
    p.add_argument("--cap", type=int, default=12, help="Max top categories overall (X)")
    p.add_argument("--weight_pop", type=float, default=0.7, help="Weight for popularity in Vibe Index")
    p.add_argument("--weight_vol", type=float, default=0.3, help="Weight for (1 - volatility) in Vibe Index")
    p.add_argument("--risk_gamma", type=float, default=1.5, help="Risk penalty curvature (gamma)")
    p.add_argument("--reward_dampen", type=float, default=0.5, help="Reward dampening by popularity [0,1]")
    p.add_argument("--outdir", type=str, default="exports", help="Base output directory in project")
    return p.parse_args()


def compute_main_items(engine: GeminiAnalysisEngine, steps: int) -> List[Dict[str, Any]]:
    class_map = engine.get_all_classes()
    items: List[Dict[str, Any]] = []
    for p_key, secs in class_map.items():
        for s_key in secs:
            ts = engine.get_monthly_series_by_class(p_key, s_key)
            if ts is None or ts.empty:
                continue
            label = f"{p_key.strip()} — {s_key.strip()}"
            ar = engine.run_full_arima_analysis_from_series(ts, label, steps=steps)
            gr = engine.run_full_garch_analysis_from_series(ts, label, steps=steps)
            series_df = ts.reset_index()
            series_df.columns = ["date", "value"]
            items.append({
                "label": label,
                "series": series_df.assign(date=series_df["date"].dt.date.astype(str)).to_dict(orient="records"),
                "arima": ar,
                "garch": gr,
            })
    return items


def compute_other_inner(engine: GeminiAnalysisEngine, steps: int, top_n: int = 100) -> List[Dict[str, Any]]:
    df = engine.df.copy()
    if "secondary_class" not in df.columns:
        return []
    df = df[df["secondary_class"].astype(str) == "other"]
    if df.empty:
        return []
    col = "canonical_tag" if "canonical_tag" in df.columns else "tag"
    inner: List[Dict[str, Any]] = []
    for inner_tag in df[col].value_counts().head(top_n).index.tolist():
        df_i = engine.df[engine.df[col].astype(str) == str(inner_tag)]
        if df_i.empty:
            continue
        ts_i = df_i.set_index("timestamp").sort_index().resample("MS").size().astype(float).asfreq("MS").fillna(0.0)
        if len(ts_i) < 8:
            continue
        label_i = f"Other: {inner_tag}"
        arima_i = engine.run_full_arima_analysis_from_series(ts_i, label_i, steps=steps)
        garch_i = engine.run_full_garch_analysis_from_series(ts_i, label_i, steps=steps)
        series_small = ts_i.reset_index()
        series_small.columns = ["date", "value"]
        pop_i = None
        vol_i = None
        try:
            f = arima_i.get("forecast", [])
            pop_i = float(sum([x.get("mean", 0.0) or 0.0 for x in f])) if f else None
        except Exception:
            pop_i = None
        try:
            vf = garch_i.get("volatility_forecast", [])
            vv = [x.get("volatility", None) for x in vf]
            vv = [float(x) for x in vv if x is not None]
            vol_i = float(sum(vv) / len(vv)) if vv else None
        except Exception:
            vol_i = None
        inner.append({
            "label": label_i,
            "series": series_small.assign(date=series_small["date"].dt.date.astype(str)).to_dict(orient="records"),
            "arima": arima_i,
            "garch": garch_i,
            "forecast_popularity": pop_i,
            "avg_volatility": vol_i,
        })
    return inner


def compute_vibe_and_top(
    main_items: List[Dict[str, Any]],
    other_inner: List[Dict[str, Any]],
    w_pop: float,
    w_vol: float,
    cap: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    rows = []
    for it in main_items:
        pop = None
        vol = None
        lbl_main = str(it.get("label") or "").strip()
        # Exclude umbrella 'Other' category from overall ranking
        if lbl_main.lower().endswith(" — other"):
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
        rows.append({"label": lbl_main, "source": "main", "forecast_popularity": pop, "avg_volatility": vol})
    for it in other_inner:
        rows.append({
            "label": it.get("label"),
            "source": "other",
            "forecast_popularity": it.get("forecast_popularity"),
            "avg_volatility": it.get("avg_volatility"),
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["forecast_popularity", "avg_volatility"], how="any")
    eps = 1e-9
    if df.empty:
        return df, [], {"pop_min": None, "pop_max": None, "vol_min": None, "vol_max": None}
    pmin, pmax = df["forecast_popularity"].min(), df["forecast_popularity"].max()
    vmin, vmax = df["avg_volatility"].min(), df["avg_volatility"].max()
    df["pop_nrm"] = (df["forecast_popularity"] - pmin) / (max(eps, pmax - pmin))
    vol_raw = (df["avg_volatility"] - vmin) / (max(eps, vmax - vmin))
    # nonlinear penalty with dampening: risk^gamma scaled down by popularity
    risk = vol_raw.pow(RISK_GAMMA)
    eff_risk = risk * (1.0 - REWARD_DAMPEN * df["pop_nrm"])
    df["vol_nrm"] = 1.0 - eff_risk
    wsum = max(eps, (w_pop + w_vol))
    df["composite_index"] = (w_pop * df["pop_nrm"] + w_vol * df["vol_nrm"]) / wsum
    df = df.sort_values("composite_index", ascending=False)
    top_labels = df.head(cap)["label"].tolist()
    norm_meta = {
        "pop_min": float(pmin) if pmin is not None else None,
        "pop_max": float(pmax) if pmax is not None else None,
        "vol_min": float(vmin) if vmin is not None else None,
        "vol_max": float(vmax) if vmax is not None else None,
        "epsilon": 1e-9,
        "weights": {"pop": float(w_pop), "vol": float(w_vol)},
        "risk_gamma": float(RISK_GAMMA),
        "reward_dampen": float(REWARD_DAMPEN),
    }
    return df, top_labels, norm_meta


def build_charts_for_label(lbl: str, series_records: List[Dict[str, Any]], ar: Dict[str, Any], gr: Dict[str, Any]) -> List[Tuple[str, bytes]]:
    outputs: List[Tuple[str, bytes]] = []
    try:
        hist_df = pd.DataFrame(series_records)
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
        png_overlay = chart_to_png_bytes(overlay)
        if png_overlay:
            outputs.append((f"charts/{sanitize_filename(lbl)}__arima.png", png_overlay))
    except Exception:
        pass
    try:
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
                outputs.append((f"charts/{sanitize_filename(lbl)}__garch_vol.png", png_vol))
    except Exception:
        pass
    return outputs


def main() -> None:
    args = build_args()
    out_base = os.path.join(os.getcwd(), args.outdir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_base, f"full_analysis_{ts}")
    charts_dir = os.path.join(out_dir, "charts")
    audit_dir = os.path.join(out_dir, "audit")
    ensure_dir(charts_dir)
    ensure_dir(audit_dir)

    # Instantiate engine with Gemini calls disabled during init
    api_key = os.environ.get("GEMINI_API_KEY", "dummy_key")
    engine = GeminiAnalysisEngine(
        api_key=api_key,
        auto_consolidate=False,
        mapping_cache_path="tag_taxonomy.json",
        enable_tag_cleaning=True,
    )

    # Main items and 'Other' subtags
    items = compute_main_items(engine, steps=args.steps)
    other_inner = compute_other_inner(engine, steps=args.steps, top_n=100)

    # Compute vibe ranking and top labels
    global RISK_GAMMA, REWARD_DAMPEN
    RISK_GAMMA = float(args.risk_gamma)
    REWARD_DAMPEN = float(args.reward_dampen)
    vibe_df, top_labels, vibe_norm_meta = compute_vibe_and_top(items, other_inner, args.weight_pop, args.weight_vol, args.cap)

    # Build top selection entries
    top_selection: List[Dict[str, Any]] = []
    for lbl in top_labels:
        it = next((x for x in items if x["label"] == lbl), None)
        inner = next((x for x in other_inner if x.get("label") == lbl), None)
        src = "main" if it is not None else "other" if inner is not None else None
        if src is None:
            continue
        rec = it if it is not None else inner
        # descriptive stats
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
        top_selection.append({
            "label": lbl,
            "source": src,
            "descriptive_statistics": stats,
            "arima_summary": {"best_order": ar.get("best_order"), "aic_score": ar.get("aic_score")},
            "garch_summary": {"alpha[1]": params.get("alpha[1]"), "beta[1]": params.get("beta[1]"), "arch_lm_pvalue": arch_test.get("lm_pvalue")},
            "series": rec.get("series", []),
            "arima_forecast": ar.get("forecast", []),
            "garch_volatility_forecast": gr.get("volatility_forecast", []),
        })

    # Build audit JSONs
    # Preprocessing metadata
    try:
        raw_rows = int(pd.read_csv(engine.data_path).shape[0])
    except Exception:
        raw_rows = None
    df_stats = {
        "data_path": engine.data_path,
        "rows_loaded": raw_rows,
        "rows_after_clean": int(engine.df.shape[0]),
        "unique_tags": int(engine.df['tag'].nunique()) if 'tag' in engine.df.columns else None,
        "has_canonical_tag": bool('canonical_tag' in engine.df.columns),
        "time_index_freq": "MS",
        "timestamp_min": engine.df['timestamp'].min().isoformat() if 'timestamp' in engine.df.columns else None,
        "timestamp_max": engine.df['timestamp'].max().isoformat() if 'timestamp' in engine.df.columns else None,
    }

    bundle = {
        "meta": {"steps": args.steps},
        "items": items,
        "other_inner_rankings": [
            {"label": r.get("label"), "forecast_popularity": r.get("forecast_popularity"), "avg_volatility": r.get("avg_volatility")}
            for r in other_inner
        ],
        "vibe_rankings_all": vibe_df.to_dict(orient="records") if not vibe_df.empty else [],
        "top_selection": top_selection,
        "preprocessing": df_stats,
        "vibe_normalization": vibe_norm_meta,
        "methodology": {
            "time_series_setup": "Monthly resample (MS), zero-fill missing months; counts per tag",
            "arima": "Grid search over p∈[0..3], d∈[0..2], q∈[0..3] minimizing AIC; ADF tests pre/post differencing; residual diagnostics (Ljung-Box, Jarque-Bera); simple holdout backtest for MAE/RMSE/MAPE",
            "garch": "GARCH(1,1) on log-returns of non-zero counts, ADF on returns; ARCH-LM test; residual diagnostics; volatility forecast",
            "vibe_index": "Min-max normalize popularity and volatility across candidates; non-linear volatility penalty vol_norm = 1 - (vol_raw^gamma × (1 - reward_dampen × pop_nrm)); composite = weighted average of pop_nrm and vol_norm",
        },
    }

    # Save audit outputs
    with open(os.path.join(audit_dir, "audit_bundle.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    with open(os.path.join(audit_dir, "vibe_rankings_all.json"), "w", encoding="utf-8") as f:
        json.dump(bundle["vibe_rankings_all"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(audit_dir, "top_selection.json"), "w", encoding="utf-8") as f:
        json.dump(top_selection, f, ensure_ascii=False, indent=2)
    # other inner rankings CSV
    if other_inner:
        df_other = pd.DataFrame(other_inner)
        if not df_other.empty:
            cols = ["label", "forecast_popularity", "avg_volatility"]
            df_other[cols].to_csv(os.path.join(audit_dir, "other_inner_rankings.csv"), index=False)

    # Save charts for each top label (with titles)
    for lbl in top_labels:
        it = next((x for x in items if x["label"] == lbl), None)
        inner = next((x for x in other_inner if x.get("label") == lbl), None)
        rec = it if it is not None else inner
        if rec is None:
            continue
        # Inject titles before export
        charts = build_charts_for_label(lbl, rec.get("series", []), rec.get("arima", {}), rec.get("garch", {}))
        for fname, data in charts:
            out_path = os.path.join(out_dir, fname)
            ensure_dir(os.path.dirname(out_path))
            with open(out_path, "wb") as f:
                f.write(data)

        # Save diagnostics tables per label for convenience (CSV)
        try:
            # ARIMA diagnostics
            ar = rec.get("arima", {}) or {}
            cand = pd.DataFrame(ar.get("candidate_models", []))
            if not cand.empty:
                cand.to_csv(os.path.join(out_dir, f"audit/arima_candidates__{sanitize_filename(lbl)}.csv"), index=False)
            # Backtest
            bt = ar.get("backtest", {})
            if bt:
                pd.DataFrame([bt]).to_csv(os.path.join(out_dir, f"audit/arima_backtest__{sanitize_filename(lbl)}.csv"), index=False)
            # GARCH diagnostics
            gr = rec.get("garch", {}) or {}
            params = pd.DataFrame([gr.get("model_params", {})])
            if not params.empty:
                params.to_csv(os.path.join(out_dir, f"audit/garch_params__{sanitize_filename(lbl)}.csv"), index=False)
            arch_lm = pd.DataFrame([gr.get("arch_effect_test", {})])
            if not arch_lm.empty:
                arch_lm.to_csv(os.path.join(out_dir, f"audit/garch_archlm__{sanitize_filename(lbl)}.csv"), index=False)
            lb = pd.DataFrame((gr.get("residual_diagnostics", {}) or {}).get("ljung_box", {})).T
            if not lb.empty:
                lb.to_csv(os.path.join(out_dir, f"audit/garch_ljungbox__{sanitize_filename(lbl)}.csv"))
            jb = pd.DataFrame([(gr.get("residual_diagnostics", {}) or {}).get("jarque_bera", {})])
            if not jb.empty:
                jb.to_csv(os.path.join(out_dir, f"audit/garch_jarquebera__{sanitize_filename(lbl)}.csv"), index=False)
        except Exception:
            pass

    # Save charts and JSONs for Top 3 'Other' even if not in top overall
    # Compute other_top3 based on other-only ranking
    other_top3_labels: List[str] = []
    if vibe_df is not None and not vibe_df.empty and "source" in vibe_df.columns:
        other_only = vibe_df[vibe_df["source"] == "other"].sort_values("composite_index", ascending=False)
        other_top3_labels = other_only.head(3)["label"].tolist()
        # Save a CSV of other-only rankings and top3
        other_only[["label", "forecast_popularity", "avg_volatility", "pop_nrm", "vol_nrm", "composite_index"]].to_csv(
            os.path.join(audit_dir, "other_rankings.csv"), index=False
        )
        other_only.head(3)[["label", "forecast_popularity", "avg_volatility", "pop_nrm", "vol_nrm", "composite_index"]].to_csv(
            os.path.join(audit_dir, "other_top3.csv"), index=False
        )

    other_top3: List[Dict[str, Any]] = []
    for lbl in other_top3_labels:
        rec = next((x for x in other_inner if x.get("label") == lbl), None)
        if rec is None:
            continue
        # descriptive stats
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
        other_top3.append({
            "label": lbl,
            "descriptive_statistics": stats,
            "arima_summary": {"best_order": ar.get("best_order"), "aic_score": ar.get("aic_score")},
            "garch_summary": {"alpha[1]": params.get("alpha[1]"), "beta[1]": params.get("beta[1]"), "arch_lm_pvalue": arch_test.get("lm_pvalue")},
            "series": rec.get("series", []),
            "arima_forecast": ar.get("forecast", []),
            "garch_volatility_forecast": gr.get("volatility_forecast", []),
        })
        # charts
        charts = build_charts_for_label(lbl, rec.get("series", []), rec.get("arima", {}), rec.get("garch", {}))
        for fname, data in charts:
            parts = os.path.splitext(fname)
            out_rel = f"charts/other_top3/{os.path.basename(parts[0])}{parts[1]}"
            out_path = os.path.join(out_dir, out_rel)
            ensure_dir(os.path.dirname(out_path))
            with open(out_path, "wb") as f:
                f.write(data)

    with open(os.path.join(audit_dir, "other_top3.json"), "w", encoding="utf-8") as f:
        json.dump(other_top3, f, ensure_ascii=False, indent=2)

    print(f"✅ Exported full analysis to: {out_dir}")


if __name__ == "__main__":
    main()


