from __future__ import annotations

import os
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import google.generativeai as genai
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


@dataclass
class SeriesInfo:
    tag: str
    start_date: Optional[pd.Timestamp]
    end_date: Optional[pd.Timestamp]
    num_observations: int


class GeminiAnalysisEngine:
    """Central analysis engine that loads data, runs ARIMA/GARCH, and
    delegates narrative/report generation to Gemini.

    Expected input CSV schema: columns `timestamp` (UNIX seconds) and `tag` (str).
    Each row is a single observation of a given tag at a timestamp.
    """

    def __init__(
        self,
        api_key: str,
        data_path: str = "tag.csv",
        tag_mapping_path: str = "tag_mapping.json",
        auto_consolidate: bool = True,
        max_tags_for_gemini: int = 8000,
        enable_tag_cleaning: bool = True,
        min_tag_count_threshold: int = 2,
        enable_genre_filtering: bool = False,
        min_genre_tag_count: int = 2,
        mapping_cache_path: str = "tag_taxonomy.json",
    ) -> None:
        if not api_key:
            raise ValueError("A valid Gemini API key is required.")

        genai.configure(api_key=api_key)
        # gemini-1.5-flash is fast and cost-effective for report generation
        self.model = genai.GenerativeModel("gemini-1.5-flash")

        self.data_path = data_path
        self.tag_mapping_path = tag_mapping_path
        self.max_tags_for_gemini = max_tags_for_gemini
        self.enable_tag_cleaning = enable_tag_cleaning
        self.min_tag_count_threshold = max(0, int(min_tag_count_threshold))
        self.enable_genre_filtering = enable_genre_filtering
        self.min_genre_tag_count = max(0, int(min_genre_tag_count))
        self.mapping_cache_path = mapping_cache_path
        self.tag_taxonomy: Dict[str, Dict[str, str]] = {}
        self.df = self._load_and_prepare_data(self.data_path)

        # Optional: consolidate variants first
        if auto_consolidate:
            try:
                self.consolidate_tags_with_gemini(
                    mapping_path=self.tag_mapping_path,
                    max_tags_for_gemini=self.max_tags_for_gemini,
                )
            except Exception as exc:
                # Fallback silently; analyses can still run on raw `tag`
                print(f"[warn] Tag consolidation skipped due to error: {exc}")

        # Clean meaningless tags prior to taxonomy to reduce noise
        if self.enable_tag_cleaning:
            self.filter_meaningless_tags(
                column="canonical_tag" if "canonical_tag" in self.df.columns else "tag",
                min_count=self.min_tag_count_threshold,
            )

        # Build a two-level taxonomy for improved UX (always attempt before any further filtering)
        # Always establish a deterministic fixed-genre taxonomy for dependable comparisons
        try:
            self.create_tag_taxonomy()
        except Exception as exc:
            print(f"[warn] Tag taxonomy generation failed: {exc}")

        # Optional: only after taxonomy, narrow dataset to genre-like if requested
        if self.enable_genre_filtering:
            self.filter_to_genre_style_tags(
                column="canonical_tag" if "canonical_tag" in self.df.columns else "tag",
                min_count=self.min_genre_tag_count,
            )

        # Build multi-label classification columns for business analysis
        try:
            self.build_multi_label_classification()
        except Exception as exc:
            print(f"[warn] Multi-label classification failed: {exc}")

    # -----------------------------
    # Data preparation
    # -----------------------------
    def _load_and_prepare_data(self, csv_path: str) -> pd.DataFrame:
        """Load CSV and prepare canonical dataframe.

        - Reads CSV
        - Coerces `timestamp` to datetime (auto-parse string or epoch)
        - Normalizes `tag` to lowercase, stripped
        - Drops invalid rows
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at: {csv_path}")

        df = pd.read_csv(csv_path)

        expected_cols = {"timestamp", "tag"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV must contain columns {expected_cols}, missing: {sorted(missing)}"
            )

        # Auto-parse timestamps whether they are human-readable strings
        # like "2013-05-10 01:41:18" or epoch seconds.
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        # Drop tz to avoid downstream far-future artifacts from tz math
        try:
            ts = ts.dt.tz_convert(None)
        except Exception:
            try:
                ts = ts.dt.tz_localize(None)
            except Exception:
                pass
        df["timestamp"] = ts
        df["tag"] = (
            df["tag"].astype(str).str.lower().str.strip()
        )

        df = df.dropna(subset=["timestamp", "tag"])  # remove invalid rows
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_all_tags(self) -> List[str]:
        """Return sorted unique tags present in the dataset."""
        column = "canonical_tag" if "canonical_tag" in self.df.columns else "tag"
        tags = self.df[column].dropna()
        tags = tags[tags.str.len() > 0]
        return sorted(tags.unique().tolist())

    def get_monthly_series(self, tag: str) -> pd.Series:
        """Return a monthly count time series (frequency = MS) for a given tag.

        The series counts the number of occurrences per month. Missing months are
        filled with zeros.
        """
        column = "canonical_tag" if "canonical_tag" in self.df.columns else "tag"
        tag_clean = str(tag).lower().strip()
        df_tag = self.df[self.df[column] == tag_clean]
        if df_tag.empty:
            return pd.Series(dtype=float)

        df_tag = df_tag.set_index("timestamp").sort_index()
        # sanitize index
        if getattr(df_tag.index, "tz", None) is not None:
            df_tag.index = df_tag.index.tz_localize(None)
        monthly_counts = df_tag.resample("MS").size().astype(float)
        # Ensure continuous monthly index and fill gaps with zeros
        monthly_counts = (
            monthly_counts.asfreq("MS").fillna(0.0)
        )
        monthly_counts.name = "count"
        return monthly_counts

    # -----------------------------
    # ARIMA analysis
    # -----------------------------
    def run_full_arima_analysis(
        self, tag: str, steps: int = 12, p_range: Tuple[int, int] = (0, 3),
        d_range: Tuple[int, int] = (0, 2), q_range: Tuple[int, int] = (0, 3)
    ) -> Dict[str, Any]:
        """Find a reasonable ARIMA(p,d,q) by AIC grid search and forecast.

        Returns a dictionary with model metadata, diagnostics, and forecast.
        """
        series = self.get_monthly_series(tag)
        series_info = self._summarize_series(tag, series)

        if series.empty or series_info.num_observations < 6:
            return {
                "model_type": "ARIMA",
                "tag": tag,
                "error": "Insufficient data for ARIMA (need at least 6 monthly observations).",
                "series_info": asdict(series_info),
            }

        best_aic = np.inf
        best_order: Optional[Tuple[int, int, int]] = None
        best_model = None

        p_values = range(p_range[0], p_range[1] + 1)
        d_values = range(d_range[0], d_range[1] + 1)
        q_values = range(q_range[0], q_range[1] + 1)

        # Simple grid search for a small hyperparameter space
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(
                            series,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = model.fit()
                        if np.isfinite(res.aic) and res.aic < best_aic:
                            best_aic = res.aic
                            best_order = (p, d, q)
                            best_model = res
                    except Exception:
                        # Skip non-invertible / non-stationary / convergence issues
                        continue

        if best_model is None or best_order is None:
            return {
                "model_type": "ARIMA",
                "tag": tag,
                "error": "ARIMA grid search failed to converge on a model.",
                "series_info": asdict(series_info),
            }

        # Forecast next `steps` months
        forecast_res = best_model.get_forecast(steps=steps)
        sf = forecast_res.summary_frame(alpha=0.05)
        sf = sf.rename(
            columns={
                "mean": "mean",
                "mean_ci_lower": "lower",
                "mean_ci_upper": "upper",
            }
        )[["mean", "lower", "upper"]]

        # Build a time index for future months aligned to monthly start
        last_idx = series.index.max()
        future_index = pd.date_range(
            start=(last_idx + pd.offsets.MonthBegin(1)), periods=steps, freq="MS"
        )
        sf.index = future_index

        result: Dict[str, Any] = {
            "model_type": "ARIMA",
            "tag": tag,
            "best_order": list(best_order),
            "aic_score": float(best_aic),
            "series_info": asdict(series_info),
            "forecast": self._frame_to_records(sf),
            "summary_table": str(best_model.summary()),
        }
        return result

    # -----------------------------
    # GARCH analysis
    # -----------------------------
    def run_full_garch_analysis(self, tag: str, steps: int = 12) -> Dict[str, Any]:
        """Fit a GARCH(1,1) model to monthly log returns of the tag counts and
        forecast volatility for `steps` months.
        """
        series = self.get_monthly_series(tag).astype(float)
        series_info = self._summarize_series(tag, series)

        if series.empty or series_info.num_observations < 8:
            return {
                "model_type": "GARCH",
                "tag": tag,
                "error": "Insufficient data for GARCH (need at least 8 monthly observations).",
                "series_info": asdict(series_info),
            }

        # Compute log returns, dropping zeros (cannot log) and NaNs
        series_nonzero = series.replace(0.0, np.nan).dropna()
        log_series = np.log(series_nonzero)
        returns = log_series.diff().dropna() * 100.0  # scale to percentage points

        if len(returns) < 6:
            return {
                "model_type": "GARCH",
                "tag": tag,
                "error": "Insufficient return observations for GARCH (need at least 6).",
                "series_info": asdict(series_info),
            }

        try:
            am = arch_model(returns, vol="Garch", p=1, o=0, q=1, dist="normal")
            garch_res = am.fit(disp="off")
        except Exception as exc:
            return {
                "model_type": "GARCH",
                "tag": tag,
                "error": f"GARCH fit failed: {exc}",
                "series_info": asdict(series_info),
            }

        forecasts = garch_res.forecast(horizon=steps, reindex=False)
        # Variance forecasts for the next `steps` at the last in-sample point
        variance_forecast = forecasts.variance.iloc[-1]
        variance_values = variance_forecast.values.astype(float)
        volatility_values = np.sqrt(variance_values)

        # Build future index based on the original count series, not returns
        last_idx = series.index.max()
        future_index = pd.date_range(
            start=(last_idx + pd.offsets.MonthBegin(1)), periods=steps, freq="MS"
        )

        vol_forecast_df = pd.DataFrame(
            {
                "variance": variance_values,
                "volatility": volatility_values,
            },
            index=future_index,
        )

        result = {
            "model_type": "GARCH",
            "tag": tag,
            "model_params": {k: float(v) for k, v in garch_res.params.items()},
            "series_info": asdict(series_info),
            "volatility_forecast": self._frame_to_records(vol_forecast_df),
            "summary_table": str(garch_res.summary()),
        }
        return result

    # -----------------------------
    # Multi-label tag classification (Genre, Tone, Style, Tropes, Setting, Source, Reference)
    # -----------------------------
    def build_multi_label_classification(self) -> None:
        """Create multi-label categories per tag and attach columns to the DataFrame.

        Columns added (JSON arrays of strings):
        - ml_genre, ml_tone, ml_style, ml_trope, ml_setting, ml_source, ml_reference
        """
        source_column = "canonical_tag" if "canonical_tag" in self.df.columns else "tag"
        tag_values = self.df[source_column].dropna().astype(str).unique().tolist()

        # Build a per-tag cache to avoid recomputation
        multi_map: Dict[str, Dict[str, List[str]]] = {}
        for raw in tag_values:
            ml = self._classify_multilabel(raw)
            multi_map[raw] = ml

        def _labels_json(tag_value: str, key: str) -> str:
            labels = multi_map.get(tag_value, {}).get(key, [])
            # ensure unique and sorted for determinism
            labels = sorted(list({str(x).strip() for x in labels if str(x).strip()}))
            return json.dumps(labels, ensure_ascii=False)

        self.df["ml_genre"] = self.df[source_column].apply(lambda t: _labels_json(t, "genre"))
        self.df["ml_tone"] = self.df[source_column].apply(lambda t: _labels_json(t, "tone"))
        self.df["ml_style"] = self.df[source_column].apply(lambda t: _labels_json(t, "style"))
        self.df["ml_trope"] = self.df[source_column].apply(lambda t: _labels_json(t, "trope"))
        self.df["ml_setting"] = self.df[source_column].apply(lambda t: _labels_json(t, "setting"))
        self.df["ml_source"] = self.df[source_column].apply(lambda t: _labels_json(t, "source"))
        self.df["ml_reference"] = self.df[source_column].apply(lambda t: _labels_json(t, "reference"))

    def _classify_multilabel(self, raw_tag: str) -> Dict[str, List[str]]:
        s = self._basic_canonicalize(raw_tag)
        labels: Dict[str, List[str]] = {
            "genre": [], "tone": [], "style": [], "trope": [], "setting": [], "source": [], "reference": []
        }

        # Genre targets from fixed set
        for g in self._canonical_genre_targets():
            if g in s:
                labels["genre"].append(g)

        # Tone / Mood
        tone_words = {
            "dark", "funny", "atmospheric", "quirky", "sad", "inspirational", "feel good", "serious",
            "bleak", "lighthearted", "suspenseful", "romantic", "humorous", "satirical", "slow burn",
            "fast paced", "terrible",
        }
        for w in tone_words:
            if w in s:
                labels["tone"].append(w)

        # Cinematic Style / Structure
        style_words = {
            "noir", "surreal", "surrealism", "nonlinear", "black and white", "b w", "slow paced",
            "found footage", "mockumentary", "arthouse", "art house",
        }
        for w in style_words:
            if w in s:
                labels["style"].append("noir" if w == "noir" else w)

        # Tropes / Themes
        trope_words = {
            "dark hero", "coming of age", "revenge", "dystopian", "time travel", "good vs evil",
            "post apocalyptic", "apocalypse", "buddy", "heist",
        }
        for w in trope_words:
            if w in s:
                labels["trope"].append(w)

        # Setting / Location
        setting_words = {
            "new york", "los angeles", "london", "paris", "tokyo", "japan", "india", "china",
            "england", "uk", "usa", "canada", "alabama", "space", "outer space", "future",
            "world war ii", "high school",
        }
        for w in setting_words:
            if w in s:
                labels["setting"].append(w)

        # Source / Origin
        source_words = {
            "british", "french", "italian", "spanish", "korean", "japanese", "bollywood", "based on a book",
            "based on book", "based on novel", "true story", "anime", "manga",
        }
        for w in source_words:
            if w in s:
                labels["source"].append(w)

        # Reference / Proper Noun (heuristic): if not clearly matched elsewhere and looks like a name/title
        if not any(labels[k] for k in ["genre", "trope", "style", "setting", "source"]):
            tokens = [t for t in s.split(" ") if t]
            if 1 <= len(tokens) <= 4 and any(c.isalpha() for c in s):
                labels["reference"].append(s)

        return labels

    def get_monthly_series_by_multi_label(self, include: Dict[str, List[str]]) -> pd.Series:
        """Aggregate monthly counts for rows that match ALL include filters across multi-label columns.

        Example include: {"genre": ["thriller"], "style": ["noir"]}
        """
        if not include:
            return pd.Series(dtype=float)
        df = self.df.copy()
        # Parse JSON arrays back to Python lists
        def _has_all(row) -> bool:
            for key, values in include.items():
                col = f"ml_{key}"
                if col not in row or pd.isna(row[col]):
                    return False
                try:
                    items = set(json.loads(row[col]))
                except Exception:
                    items = set()
                if not set(v.strip().lower() for v in values) & set(x.strip().lower() for x in items):
                    return False
            return True

        mask = df.apply(_has_all, axis=1)
        df_f = df[mask]
        if df_f.empty:
            return pd.Series(dtype=float)
        series = df_f.set_index("timestamp").resample("MS").size().astype(float)
        series = series.asfreq("MS").fillna(0.0)
        series.name = "count"
        return series

    # -----------------------------
    # Tag taxonomy via Gemini (Primary/Secondary classes)
    # -----------------------------
    def create_tag_taxonomy(self) -> None:
        """Classify tags into a two-level hierarchy using Gemini and cache to disk."""
        # Load cache if exists
        if os.path.exists(self.mapping_cache_path):
            print("Loading tag taxonomy map from cache...")
            with open(self.mapping_cache_path, "r", encoding="utf-8") as f:
                self.tag_taxonomy = json.load(f)
        else:
            print("Generating new tag taxonomy with Gemini (this may take a moment)...")
            tag_counts = self.df["tag"].value_counts()
            tags_to_classify = tag_counts[tag_counts > 25].index.tolist()

            primary_classes = [
                "Genre", "Theme", "Tone", "Plot Device", "Character",
                "Setting", "Critique", "Production", "Unclassifiable",
            ]

            def build_prompt(batch: List[str]) -> str:
                primary_classes_str = ", ".join(primary_classes)
                return (
                    "You are an expert movie librarian and data taxonomist. Classify raw user tags "
                    "into a structured two-level hierarchy.\n\n"
                    "For each tag, do the following:\n"
                    f"1) Assign a Primary Class from this list: {primary_classes_str}.\n"
                    "2) Create a clean, consolidated Secondary Class name that groups related concepts.\n"
                    "Use the MovieLens tag-count context: these are user-applied tags on movies, not viewership.\n"
                    "If a tag is meaningless/junk, classify it as Unclassifiable.\n\n"
                    "Output MUST be NDJSON (one compact JSON per line), where each line has keys: \"tag\", \"primary\", \"secondary\".\n"
                    "Do not include prose or code fences. Output NDJSON lines only.\n\n"
                    "Example lines:\n"
                    "{\"tag\": \"superhero\", \"primary\": \"Genre\", \"secondary\": \"superhero\"}\n"
                    "{\"tag\": \"dystopian\", \"primary\": \"Theme\", \"secondary\": \"dystopian\"}\n\n"
                    f"Classify these tags (high-frequency subset):\n{batch}\n"
                )

            taxonomy: Dict[str, Dict[str, str]] = {}
            try:
                batch_size = 200
                for start in range(0, len(tags_to_classify), batch_size):
                    batch = tags_to_classify[start : start + batch_size]
                    prompt = build_prompt(batch)
                    response = self.model.generate_content(prompt)
                    text = getattr(response, "text", "") or ""
                    cleaned = text.replace("```json", "").replace("```", "").strip()
                    for line in cleaned.splitlines():
                        line = line.strip()
                        if not (line.startswith("{") and line.endswith("}")):
                            continue
                        try:
                            obj = json.loads(line)
                            tag = str(obj.get("tag", "")).strip()
                            primary = str(obj.get("primary", "")).strip()
                            secondary = str(obj.get("secondary", "")).strip()
                            if tag:
                                taxonomy[tag] = {"primary": primary or "Unclassifiable", "secondary": secondary or "unknown"}
                        except Exception:
                            continue
                self.tag_taxonomy = taxonomy
                with open(self.mapping_cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.tag_taxonomy, f, ensure_ascii=False, indent=2)
                print(f"✅ New taxonomy map saved to {self.mapping_cache_path}")
            except Exception as exc:
                print(f"❌ Error generating taxonomy from Gemini: {exc}")
                self.tag_taxonomy = {}

        # Heuristic fallback if taxonomy is empty or contains no useful classes
        if not self.tag_taxonomy:
            print("[info] Falling back to heuristic taxonomy rules (no Gemini taxonomy parsed)")
            raw_tags = self.df["tag"].dropna().astype(str).unique().tolist()
            self.tag_taxonomy = self._build_heuristic_taxonomy(raw_tags)
            try:
                with open(self.mapping_cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.tag_taxonomy, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                print(f"[warn] Failed to save heuristic taxonomy: {exc}")

        def get_class(tag: str, level: str) -> str:
            try:
                return self.tag_taxonomy.get(tag, {}).get(level, "Unclassifiable")
            except Exception:
                return "Unclassifiable"

        self.df["primary_class"] = self.df["tag"].apply(lambda t: get_class(t, "primary"))
        self.df["secondary_class"] = self.df["tag"].apply(lambda t: get_class(t, "secondary"))
        print("✅ Tag taxonomy applied to DataFrame.")

    # Build a taxonomy without Gemini using rules/keywords
    def _build_heuristic_taxonomy(self, tags: List[str]) -> Dict[str, Dict[str, str]]:
        # Fixed, comparable genre set (~15-18) for reliable aggregation
        canonical_genres = [
            "action", "adventure", "animation", "comedy", "crime", "drama",
            "fantasy", "historical", "horror", "mystery", "romance",
            "romantic comedy", "sci fi", "thriller", "war", "western",
            "superhero", "documentary",
        ]

        mapping: Dict[str, Dict[str, str]] = {}
        for raw in tags:
            genre, normalized = self._map_to_fixed_genre(raw, canonical_genres)
            mapping[raw] = {"primary": "Genre", "secondary": genre or normalized}
        return mapping

    def _classify_primary_secondary(self, raw_tag: str) -> Tuple[str, str]:
        s = self._basic_canonicalize(raw_tag)
        # Keyword buckets
        theme_words = {
            "dystopian", "utopian", "revenge", "redemption", "coming of age", "survival",
            "friendship", "love triangle", "forbidden love", "war on terror", "post apocalyptic",
            "apocalypse", "time travel", "road movie", "buddy",
        }
        tone_words = {
            "dark", "lighthearted", "gritty", "campy", "whimsical", "bleak", "suspenseful",
            "romantic", "humorous", "satirical", "slow burn", "fast paced", "feel good",
        }
        plot_device_words = {
            "time travel", "amnesia", "memory loss", "flashback", "nonlinear", "found footage",
            "heist", "whodunit",
        }
        production_words = {
            "cinematography", "black and white", "b w", "soundtrack", "score", "direction",
            "screenplay", "script", "editing", "vfx", "cgi", "special effects", "3d", "imax",
        }
        critique_words_pos = {"masterpiece", "must see", "favorite", "underrated", "great", "amazing"}
        critique_words_neg = {"boring", "awful", "bad", "overrated", "trash"}
        setting_words = {
            "new york", "los angeles", "london", "paris", "tokyo", "japan", "india", "china",
            "england", "uk", "usa", "canada", "alabama", "new zealand", "space", "outer space",
        }

        # Decide primary
        if self._is_genre_like(s):
            return "Genre", s
        if any(w in s for w in theme_words):
            return "Theme", s
        if any(w in s for w in tone_words):
            return "Tone", s
        if any(w in s for w in plot_device_words):
            return "Plot Device", s
        if any(w in s for w in production_words):
            return "Production", s
        if any(w in s for w in setting_words):
            return "Setting", s
        if s in critique_words_pos or s in critique_words_neg:
            return "Critique", s

        # Person-like heuristic for Character (fallback):
        tokens = [t for t in s.split(" ") if t]
        if 1 <= len(tokens) <= 3 and all(re.fullmatch(r"[a-z]+", t) for t in tokens):
            # If not matched anywhere else, assume Character for multi-token proper-like names
            if len(tokens) >= 2:
                return "Character", s

        # Default
        return "Genre", s

    def _map_to_fixed_genre(self, raw_tag: str, targets: List[str]) -> Tuple[str, str]:
        s = self._basic_canonicalize(raw_tag)
        # High-confidence direct mappings
        direct = {
            "scifi": "sci fi",
            "science fiction": "sci fi",
            "sf": "sci fi",
            "romcom": "romantic comedy",
            "rom com": "romantic comedy",
            "docu": "documentary",
            "anime": "animation",
            "kids": "family",  # family not in fixed targets, keep animation/comedy instead
            "super hero": "superhero",
            "superheroes": "superhero",
            "super heros": "superhero",
        }
        if s in direct:
            g = direct[s]
            if g in targets:
                return g, s

        # Root-based inferences
        root_map = {
            "adventur": "adventure",
            "superhero": "superhero",
            "comed": "comedy",
            "romant": "romance",
            "thrill": "thriller",
            "myster": "mystery",
            "horr": "horror",
            "anim": "animation",
            "fantas": "fantasy",
            "war": "war",
            "western": "western",
            "crime": "crime",
            "doc": "documentary",
            "sci fi": "sci fi",
            "science": "sci fi",
            "space": "sci fi",
            "noir": "crime",
        }
        for root, g in root_map.items():
            if root in s and g in targets:
                return g, s

        # Multi-word cues like "action comedy" prioritize the first genre token found in targets
        tokens = s.split()
        for t in tokens:
            if t in targets:
                return t, s

        # Special handling: if token combination indicates superhero variants
        if "super" in tokens and ("hero" in tokens or "heroes" in tokens or "heros" in tokens):
            if "superhero" in targets:
                return "superhero", s

        # Fallback to closest target by substring similarity
        for g in targets:
            if g in s or s in g:
                return g, s

        # Default to an explicit Other bucket to avoid skewing Genre counts
        return "Other", "other"

    def get_all_classes(self) -> Dict[str, List[str]]:
        """Return mapping of primary class -> sorted list of secondary classes (excluding Unclassifiable)."""
        if "primary_class" not in self.df.columns or "secondary_class" not in self.df.columns:
            return {}
        class_map: Dict[str, List[str]] = {}
        primary_options = sorted(self.df["primary_class"].dropna().unique().tolist())
        for p in primary_options:
            if p == "Unclassifiable":
                continue
            secs = (
                self.df[self.df["primary_class"] == p]["secondary_class"].dropna().unique().tolist()
            )
            class_map[p] = sorted(secs)
        return class_map

    def get_monthly_series_by_class(self, primary_class: str, secondary_class: str) -> pd.Series:
        """Aggregate monthly counts for rows within primary_class where secondary_class term appears."""
        if "primary_class" not in self.df.columns or "secondary_class" not in self.df.columns:
            return pd.Series(dtype=float)
        df_primary = self.df[self.df["primary_class"] == primary_class]
        if df_primary.empty:
            return pd.Series(dtype=float)
        mask = df_primary["secondary_class"].astype(str).str.contains(str(secondary_class), na=False)
        df_filtered = df_primary[mask]
        if df_filtered.empty:
            return pd.Series(dtype=float)
        idxed = df_filtered.set_index("timestamp").sort_index()
        if getattr(idxed.index, "tz", None) is not None:
            idxed.index = idxed.index.tz_localize(None)
        monthly = idxed.resample("MS").size().astype(float)
        monthly = monthly.asfreq("MS").fillna(0.0)
        monthly.name = "count"
        return monthly

    # Convenience: run models directly from a provided series
    def run_full_arima_analysis_from_series(self, series: pd.Series, name: str, steps: int = 12) -> Dict[str, Any]:
        if series is None or series.empty or len(series) < 6:
            return {
                "model_type": "ARIMA",
                "tag": name,
                "error": "Insufficient data for ARIMA (need at least 6 monthly observations).",
            }
        # Stationarity check on original series
        try:
            adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series.dropna(), autolag="AIC")
            adf_result = {"statistic": float(adf_stat), "pvalue": float(adf_p), "crit": {k: float(v) for k, v in adf_crit.items()}}
        except Exception:
            adf_result = {"statistic": None, "pvalue": None, "crit": {}}

        best_aic = np.inf
        best_order: Optional[Tuple[int, int, int]] = None
        best_model = None
        candidates: List[Dict[str, Any]] = []
        p_bounds, d_bounds, q_bounds = (0, 3), (0, 2), (0, 3)
        for p in range(p_bounds[0], p_bounds[1] + 1):
            for d in range(d_bounds[0], d_bounds[1] + 1):
                for q in range(q_bounds[0], q_bounds[1] + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                        res = model.fit()
                        cand = {"order": [p, d, q], "aic": float(getattr(res, "aic", np.inf)), "bic": float(getattr(res, "bic", np.inf))}
                        candidates.append(cand)
                        if np.isfinite(res.aic) and res.aic < best_aic:
                            best_aic = res.aic
                            best_order = (p, d, q)
                            best_model = res
                    except Exception:
                        continue
        if best_model is None or best_order is None:
            return {"model_type": "ARIMA", "tag": name, "error": "ARIMA grid search failed."}
        # ADF on differenced series according to selected d
        try:
            d = best_order[1]
            s_diff = series.copy().dropna()
            for _ in range(d):
                s_diff = s_diff.diff().dropna()
            adf_d_stat, adf_d_p, _, _, adf_d_crit, _ = adfuller(s_diff, autolag="AIC")
            adf_diff_result = {"statistic": float(adf_d_stat), "pvalue": float(adf_d_p), "crit": {k: float(v) for k, v in adf_d_crit.items()}, "d": int(d)}
        except Exception:
            adf_diff_result = {"statistic": None, "pvalue": None, "crit": {}, "d": int(best_order[1])}
        forecast_res = best_model.get_forecast(steps=steps)
        sf = forecast_res.summary_frame(alpha=0.05).rename(columns={"mean": "mean", "mean_ci_lower": "lower", "mean_ci_upper": "upper"})[["mean", "lower", "upper"]]
        last_idx = series.index.max()
        future_index = pd.date_range(start=(last_idx + pd.offsets.MonthBegin(1)), periods=steps, freq="MS")
        sf.index = future_index
        # Residual diagnostics
        resid = pd.Series(getattr(best_model, "resid", pd.Series(dtype=float))).dropna()
        try:
            lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
            lb_result = {int(k): {"stat": float(lb.loc[k, "lb_stat"]), "pvalue": float(lb.loc[k, "lb_pvalue"])} for k in lb.index}
        except Exception:
            lb_result = {}
        try:
            jb_stat, jb_pvalue, _, _ = jarque_bera(resid)
            jb_result = {"stat": float(jb_stat), "pvalue": float(jb_pvalue)}
        except Exception:
            jb_result = {"stat": None, "pvalue": None}
        # Simple holdout backtest for transparency
        backtest: Dict[str, Any] = {}
        try:
            if len(series) >= 18:
                h = max(3, min(6, len(series) // 5))
                train = series.iloc[:-h]
                test = series.iloc[-h:]
                bt_model = ARIMA(train, order=best_order, enforce_stationarity=False, enforce_invertibility=False).fit()
                bt_fc = bt_model.get_forecast(steps=h).summary_frame(alpha=0.05)
                pred = bt_fc["mean"].astype(float)
                pred.index = test.index
                err = (pred - test.astype(float))
                mae = float(np.abs(err).mean())
                rmse = float(np.sqrt((err ** 2).mean()))
                eps = 1e-9
                mape = float((np.abs(err) / np.maximum(eps, np.abs(test))).mean() * 100.0)
                backtest = {"h": int(h), "mae": mae, "rmse": rmse, "mape_percent": mape, "train_end": train.index.max().date().isoformat()}
        except Exception:
            backtest = {}
        return {
            "model_type": "ARIMA",
            "tag": name,
            "best_order": list(best_order),
            "aic_score": float(best_aic),
            "candidate_models": sorted(candidates, key=lambda x: x["aic"])[:10],
            "stationarity": {"adf_original": adf_result, "adf_after_d": adf_diff_result},
            "residual_diagnostics": {"ljung_box": lb_result, "jarque_bera": jb_result},
            "data_coverage": {"start": series.index.min().date().isoformat() if len(series) else None, "end": series.index.max().date().isoformat() if len(series) else None},
            "search_space": {"p": list(range(p_bounds[0], p_bounds[1] + 1)), "d": list(range(d_bounds[0], d_bounds[1] + 1)), "q": list(range(q_bounds[0], q_bounds[1] + 1))},
            "methodology_notes": {
                "stationarity_test": "ADF on original series; if non-stationary, differencing according to d in best (p,d,q) and ADF retested.",
                "model_selection": "Grid search on small (p,d,q) ranges using AIC minimization.",
                "forecast": "Point forecast with 95% confidence intervals via statsmodels get_forecast." ,
            },
            "backtest": backtest,
            "forecast": self._frame_to_records(sf),
            "summary_table": str(best_model.summary()),
        }

    def run_full_garch_analysis_from_series(self, series: pd.Series, name: str, steps: int = 12) -> Dict[str, Any]:
        if series is None or series.empty or len(series) < 8:
            return {
                "model_type": "GARCH",
                "tag": name,
                "error": "Insufficient data for GARCH (need at least 8 monthly observations).",
            }
        s = series.astype(float).replace(0.0, np.nan).dropna()
        log_series = np.log(s)
        returns = log_series.diff().dropna() * 100.0
        if len(returns) < 6:
            return {"model_type": "GARCH", "tag": name, "error": "Insufficient return observations for GARCH (need at least 6)."}
        # Stationarity of returns
        try:
            adf_stat, adf_p, _, _, adf_crit, _ = adfuller(returns, autolag="AIC")
            adf_result = {"statistic": float(adf_stat), "pvalue": float(adf_p), "crit": {k: float(v) for k, v in adf_crit.items()}}
        except Exception:
            adf_result = {"statistic": None, "pvalue": None, "crit": {}}
        # ARCH effect test on returns pre-fit
        try:
            lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns, nlags=12)
            arch_lm = {"lm_stat": float(lm_stat), "lm_pvalue": float(lm_pvalue), "f_stat": float(f_stat), "f_pvalue": float(f_pvalue), "nlags": 12}
        except Exception:
            arch_lm = {}
        try:
            am = arch_model(returns, vol="Garch", p=1, o=0, q=1, dist="normal")
            garch_res = am.fit(disp="off")
        except Exception as exc:
            return {"model_type": "GARCH", "tag": name, "error": f"GARCH fit failed: {exc}"}
        forecasts = garch_res.forecast(horizon=steps, reindex=False)
        variance_values = forecasts.variance.iloc[-1].values.astype(float)
        volatility_values = np.sqrt(variance_values)
        last_idx = series.index.max()
        future_index = pd.date_range(start=(last_idx + pd.offsets.MonthBegin(1)), periods=steps, freq="MS")
        vol_forecast_df = pd.DataFrame({"variance": variance_values, "volatility": volatility_values}, index=future_index)
        # Residual diagnostics on standardized residuals
        try:
            std_resid = garch_res.std_resid.dropna()
            lb = acorr_ljungbox(std_resid, lags=[12, 24], return_df=True)
            lb_result = {int(k): {"stat": float(lb.loc[k, "lb_stat"]), "pvalue": float(lb.loc[k, "lb_pvalue"])} for k in lb.index}
        except Exception:
            lb_result = {}
        try:
            jb_stat, jb_pvalue, _, _ = jarque_bera(std_resid)
            jb_result = {"stat": float(jb_stat), "pvalue": float(jb_pvalue)}
        except Exception:
            jb_result = {"stat": None, "pvalue": None}
        return {
            "model_type": "GARCH",
            "tag": name,
            "model_params": {k: float(v) for k, v in garch_res.params.items()},
            "stationarity": {"adf_returns": adf_result},
            "arch_effect_test": arch_lm,
            "residual_diagnostics": {"ljung_box": lb_result, "jarque_bera": jb_result},
            "methodology_notes": {
                "returns_construction": "Log returns of non-zero monthly counts, scaled by 100.",
                "model": "GARCH(1,1) with normal innovations; variance and volatility forecast for horizon.",
            },
            "volatility_forecast": self._frame_to_records(vol_forecast_df),
            "summary_table": str(garch_res.summary()),
        }

    # -----------------------------
    # Tag cleaning (heuristics + frequency threshold)
    # -----------------------------
    def filter_meaningless_tags(self, column: str = "tag", min_count: int = 2) -> None:
        """Remove rows whose tag value in `column` appears meaningless by heuristics
        or occurs fewer than `min_count` times overall.

        Heuristics (case-insensitive, applied post-normalization):
        - Keep if contains at least one alphabetic character a-z
        - Else if all digits:
            - Keep specific whitelist like "007"
            - Otherwise drop if length >= 4; allow <= 3 to capture short numeric codes if desired
        - Drop if final token length is 1 character (e.g., "a")
        """
        if column not in self.df.columns:
            return

        normalized = self.df[column].astype(str).str.lower().str.strip()
        mask_meaningful = normalized.apply(self._is_meaningful_tag)

        # Apply frequency threshold on the chosen column
        counts = normalized[mask_meaningful].value_counts()
        allowed = set(counts[counts >= max(1, int(min_count))].index)
        mask_freq = normalized.apply(lambda x: x in allowed)

        mask_final = mask_meaningful & mask_freq
        before = len(self.df)
        self.df = self.df[mask_final].copy()
        self.df.reset_index(drop=True, inplace=True)
        after = len(self.df)
        print(f"[info] Tag cleaning reduced rows from {before} to {after} (column={column}).")

    def _is_meaningful_tag(self, tag: str) -> bool:
        s = str(tag).lower().strip()
        if not s:
            return False
        # reject single-character tags
        if len(s) == 1:
            return False
        # contains alphabetic character
        if re.search(r"[a-z]", s):
            return True
        # allow specific digit-only whitelists
        whitelist = {"007"}
        if s in whitelist:
            return True
        # reject long digit-only strings (e.g., 010205)
        if re.fullmatch(r"\d+", s):
            return len(s) <= 3
        # by default, keep
        return True

    # -----------------------------
    # Keep only genre/style-like tags
    # -----------------------------
    def filter_to_genre_style_tags(self, column: str = "tag", min_count: Optional[int] = None) -> None:
        if column not in self.df.columns:
            return
        if min_count is None:
            min_count = self.min_genre_tag_count

        normalized = self.df[column].astype(str).str.lower().str.strip()
        mask_genre = normalized.apply(self._is_genre_like)

        counts = normalized[mask_genre].value_counts()
        allowed = set(counts[counts >= max(1, int(min_count))].index)
        mask_freq = normalized.apply(lambda x: x in allowed)

        mask_final = mask_genre & mask_freq
        before = len(self.df)
        self.df = self.df[mask_final].copy()
        self.df.reset_index(drop=True, inplace=True)
        after = len(self.df)
        print(f"[info] Genre/style filter reduced rows from {before} to {after} (column={column}).")

    def _is_genre_like(self, tag: str) -> bool:
        s = str(tag).lower().strip()
        if not s:
            return False

        # Core genre/style vocabulary (expandable)
        genre_keywords = {
            "action", "adventure", "comedy", "romance", "romantic", "drama", "thriller",
            "horror", "mystery", "crime", "noir", "film noir", "sci fi", "science fiction",
            "fantasy", "animation", "anime", "documentary", "docu", "family", "kids",
            "war", "western", "biopic", "biographical", "historical", "period", "satire",
            "parody", "spoof", "slasher", "zombie", "vampire", "superhero", "dystopian",
            "utopian", "post apocalyptic", "apocalypse", "time travel", "teen", "coming of age",
            "sports", "sport", "musical", "musicals", "spy", "espionage", "heist", "gangster",
            "mafia", "courtroom", "legal", "medical", "supernatural", "occult", "ghost",
            "space", "alien", "cyberpunk", "steampunk", "martial arts", "kung fu", "samurai",
            "ninja", "arthouse", "art house", "indie", "experimental", "bollywood", "kollywood",
            "tollywood", "kdrama", "melodrama", "psychological", "erotic", "dark comedy",
            "black comedy", "romcom", "road movie", "buddy", "cult", "giallo", "splatter",
            "found footage", "mockumentary",
        }

        # Exact match or substring match for multi-word phrases
        for kw in genre_keywords:
            if kw == s:
                return True
            if kw in s:
                return True

        # Pattern-based: tags like "political drama", "legal thriller", "action comedy"
        genre_heads = {
            "drama", "thriller", "comedy", "horror", "romance", "action", "adventure", "mystery",
            "crime", "fantasy", "documentary", "animation", "western", "musical",
        }
        tokens = s.split()
        if len(tokens) >= 2:
            if tokens[-1] in genre_heads or tokens[0] in genre_heads:
                return True
        # Common suffixes indicating style
        if s.endswith(" film") or s.endswith(" movie"):
            return True

        return False

    # -----------------------------
    # AI-powered tag consolidation
    # -----------------------------
    def consolidate_tags_with_gemini(
        self,
        mapping_path: str = "tag_mapping.json",
        force: bool = False,
        max_tags_for_gemini: Optional[int] = None,
        use_canonical_targets: bool = True,
    ) -> Dict[str, str]:
        """Create or load a mapping from original tag -> canonical tag using Gemini.

        - Loads cached mapping if available and sufficiently covers current tags
        - Otherwise, asks Gemini to consolidate and saves the mapping
        - Applies the mapping to create `canonical_tag` column
        """
        if max_tags_for_gemini is None:
            max_tags_for_gemini = self.max_tags_for_gemini

        unique_tags = self.df["tag"].dropna().astype(str).str.strip()
        unique_tags = unique_tags[unique_tags.str.len() > 0]
        unique_tags_list = sorted(unique_tags.unique().tolist())

        # Attempt to load existing mapping
        cached_mapping = self._load_tag_mapping(mapping_path)
        if cached_mapping and not force:
            coverage = sum(1 for t in unique_tags_list if t in cached_mapping) / max(1, len(unique_tags_list))
            if coverage >= 0.90:  # Accept mapping if it covers at least 90% of tags
                mapping = self._sanitize_mapping(cached_mapping)
                self._apply_mapping(mapping)
                return mapping

        # Build request payload for Gemini
        counts = Counter(self.df["tag"].dropna().astype(str))
        # If too many tags, keep the most frequent subset for Gemini; others fallback to heuristic
        if len(unique_tags_list) > max_tags_for_gemini:
            top_tags = [t for t, _ in counts.most_common(max_tags_for_gemini)]
            remaining_tags = [t for t in unique_tags_list if t not in top_tags]
        else:
            top_tags = unique_tags_list
            remaining_tags = []

        tag_counts_payload = {t: int(counts[t]) for t in top_tags}

        canonical_targets = self._canonical_genre_targets() if use_canonical_targets else []

        canonical_policy = (
            "Canonicalization policy: lowercase; strip leading/trailing spaces; "
            "replace hyphens with spaces; remove punctuation except spaces and digits; "
            "collapse multiple spaces to one; singularize simple English plurals by removing a trailing 's' "
            "when the word is longer than 3 letters and does not end with 'ss', 'us', or 'is'. "
            "When mapping synonyms (e.g., 'sci-fi' and 'science fiction'), choose as canonical the variant "
            "that appears most frequently in the provided tag_counts; if tied, choose the lexicographically smallest."
        )

        prompt = (
            "You are an expert data librarian and ontologist. Group semantically similar tags "
            "(plurals, spelling variations, hyphenation, synonyms) under a single canonical tag.\n\n"
            f"{canonical_policy}\n\n"
            "Return ONLY a JSON object mapping every original tag EXACTLY as provided to a canonical tag string.\n"
            "Do not include explanations or code fences.\n\n"
            "Example: {\n  \"superheros\": \"superhero\",\n  \"super-hero\": \"superhero\",\n  \"sci-fi\": \"sci fi\"\n}\n\n"
            + ("When appropriate, map tags to one of these canonical genre/style targets: "
               + ", ".join(sorted(canonical_targets)) + ".\n\n" if canonical_targets else "")
            + "Here is a JSON object with tag_counts (original_tag -> frequency):\n"
            f"{json.dumps(tag_counts_payload)[:250000]}\n\n"  # safeguard extremely large
            "Produce a JSON mapping for ALL keys present above."
        )

        try:
            response = self.model.generate_content(prompt)
            text = getattr(response, "text", "") or ""
            mapping = self._parse_json_strict(text)
        except Exception as exc:
            print(f"[warn] Gemini mapping generation failed: {exc}")
            mapping = {}

        # Fallbacks and completion for missing entries
        mapping = self._sanitize_mapping(mapping)
        for t in top_tags:
            if t not in mapping:
                mapping[t] = self._canonical_by_rules(t, canonical_targets) or self._basic_canonicalize(t)
        for t in remaining_tags:
            mapping[t] = self._canonical_by_rules(t, canonical_targets) or self._basic_canonicalize(t)

        # Apply and persist
        self._apply_mapping(mapping)
        self._save_tag_mapping(mapping_path, mapping)
        return mapping

    # -----------------------------
    # Consolidation helpers
    # -----------------------------
    def _apply_mapping(self, mapping: Dict[str, str]) -> None:
        col = "canonical_tag"
        self.df[col] = self.df["tag"].map(mapping).fillna(self.df["tag"].map(self._basic_canonicalize))
        self.df[col] = self.df[col].astype(str).str.lower().str.strip()

    def _load_tag_mapping(self, path: str) -> Optional[Dict[str, str]]:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            print(f"[warn] Failed to load tag mapping: {exc}")
        return None

    def _save_tag_mapping(self, path: str, mapping: Dict[str, str]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[warn] Failed to save tag mapping: {exc}")

    def _sanitize_mapping(self, mapping: Dict[str, str]) -> Dict[str, str]:
        clean: Dict[str, str] = {}
        for k, v in (mapping or {}).items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, str):
                v = str(v)
            k_clean = k.strip().lower()
            v_clean = v.strip().lower()
            if not k_clean:
                continue
            if not v_clean:
                v_clean = self._basic_canonicalize(k_clean)
            clean[k_clean] = v_clean
        return clean

    def _basic_canonicalize(self, tag: str) -> str:
        s = str(tag).lower().strip()
        s = s.replace("-", " ")
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        # naive plural -> singular for the last token
        parts = s.split(" ")
        if parts:
            last = parts[-1]
            if len(last) > 3 and last.endswith("s") and not (last.endswith("ss") or last.endswith("us") or last.endswith("is")):
                parts[-1] = last[:-1]
            s = " ".join(parts)
        return s

    def _canonical_by_rules(self, tag: str, canonical_targets: List[str]) -> Optional[str]:
        """Rule-based canonicalization toward a fixed set of genre/style targets.

        Examples:
        - "adventure boat" -> "adventure"
        - "adventurous" -> "adventure"
        - "sci-fi" -> "sci fi"
        - "romcom" -> "romantic comedy"
        """
        s = self._basic_canonicalize(tag)

        # Synonym dictionary (extend as needed)
        synonym_map = {
            "scifi": "sci fi",
            "sci fi": "sci fi",
            "science fiction": "sci fi",
            "sf": "sci fi",
            "romcom": "romantic comedy",
            "rom com": "romantic comedy",
            "romantic comedy": "romantic comedy",
            "arthouse": "arthouse",
            "art house": "arthouse",
            "docu": "documentary",
            "anime": "animation",
            "kids": "family",
            "children": "family",
        }
        if s in synonym_map:
            return synonym_map[s]

        # Morphological roots to canonical
        root_map = {
            "adventur": "adventure",
            "comed": "comedy",
            "romant": "romance",
            "thrill": "thriller",
            "myster": "mystery",
            "horr": "horror",
            "anim": "animation",
            "doc": "documentary",
            "fantas": "fantasy",
        }
        for root, canonical in root_map.items():
            if root in s:
                return canonical

        # Token/substring match against canonical targets
        for target in sorted(canonical_targets or self._canonical_genre_targets(), key=len, reverse=True):
            if target == s:
                return target
            if target in s:
                return target

        return None

    def _canonical_genre_targets(self) -> List[str]:
        # Align with genre/style filter vocabulary; keep concise canonical set
        return sorted({
            "action", "adventure", "animation", "anime", "biopic", "comedy", "crime",
            "documentary", "drama", "fantasy", "historical", "horror", "mystery",
            "noir", "romance", "romantic comedy", "sci fi", "sports", "thriller",
            "war", "western", "family", "musical", "superhero", "dystopian",
            "post apocalyptic", "spy", "heist", "gangster", "teen", "coming of age",
            "psychological", "dark comedy", "black comedy", "cult", "indie", "arthouse",
        })

    def _parse_json_strict(self, text: str) -> Dict[str, str]:
        # Remove code fences if present
        cleaned = text.strip()
        cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            # Try to extract first JSON object substring
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except Exception:
                    pass
        raise ValueError("Failed to parse JSON from Gemini response")

    # -----------------------------
    # Report generation via Gemini
    # -----------------------------
    def generate_report(self, analysis_results: List[Dict[str, Any]], user_query: str) -> str:
        """Compose a prompt with analysis artifacts and ask Gemini to produce a
        comprehensive business-focused narrative.
        """
        # Keep payload compact by ensuring any DataFrames are already serialized
        prompt = (
            "You are an expert data science consultant helping a student write a final term paper. "
            "Interpret the results of ARIMA and GARCH models and produce a business-focused report.\n\n"
            f"User Query: {user_query}\n\n"
            "Critical data context: The dataset is from MovieLens tag activity. The key metric is the count of user-applied tags over time, not viewership or search interest. "
            "Interpret observed trends as how frequently users labeled movies with each tag (a measure of salience/signifier strength in user minds). Do not claim viewership.\n\n"
            "Provided Data and Model Outputs (JSON-like):\n"
            f"{analysis_results}\n\n"
            "Instructions: Generate a structured, specific report with these sections:\n"
            "1) Executive Summary: A concise answer to the user's question.\n"
            "2) Context & Data Definition: Explain the tag-count nature of the data.\n"
            "3) Model Insights: Interpret ARIMA forecasts and GARCH volatility concisely (mention AIC/order).\n"
            "4) Volatility Drivers (Hypotheses): If actor tags are spiky (e.g., 'al pacino'), speculate on plausible real-world causes (new releases, awards, news) and contrast with stable location tags (e.g., 'alabama').\n"
            "5) Strategic Recommendations: For a new streaming service in Kamloops, BC, provide three specific, creative actions PER TAG. "
            "   - For high-buzz tags, think partnerships (e.g., local film festival retrospectives), editorial features, or low-cost content strategies (e.g., essays, curation).\n"
            "   - For stable niche tags, suggest targeted, low-cost acquisitions or programming to signal depth in that niche.\n"
            "6) Business Decision: Make a call grounded in risk/return trade-offs evidenced by the models.\n"
            "7) Methodology (Layperson): Briefly explain ARIMA and GARCH in simple terms.\n"
            "8) Limitations & Next Steps: Note data limitations (tag counts vs viewership) and validation ideas.\n\n"
            "Style: Use concise, professional language. Be concrete and locally relevant to Kamloops, BC. Include small tables when helpful."
        )

        response = self.model.generate_content(prompt)
        return getattr(response, "text", "[No response text returned by Gemini]")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _frame_to_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert a DataFrame with a DatetimeIndex to list-of-dicts with ISO dates."""
        records: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            item: Dict[str, Any] = {"date": pd.Timestamp(idx).date().isoformat()}
            for col, val in row.items():
                try:
                    item[col] = float(val)
                except Exception:
                    item[col] = None
            records.append(item)
        return records

    def _summarize_series(self, tag: str, series: pd.Series) -> SeriesInfo:
        if series.empty:
            return SeriesInfo(tag=tag, start_date=None, end_date=None, num_observations=0)
        return SeriesInfo(
            tag=tag,
            start_date=pd.Timestamp(series.index.min()),
            end_date=pd.Timestamp(series.index.max()),
            num_observations=int(series.shape[0]),
        )


