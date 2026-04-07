"""
Phase 4: Regression Analysis — Migration Determinants
=======================================================
Nodes and Nations: A Complex Network Study of Global Migration

Steps:
  1. Merge centrality metrics with supplementary factors panel
  2. Fit Baseline (3-factor) OLS model: Log(Pop), Log(GDP), Conflict
  3. Fit Full (7-factor) OLS model: all seven predictors
  4. Run VIF analysis on all predictors
  5. Fit separate origin-side and destination-side models
  6. Export coefficient tables, model comparison, VIF results

Inputs:  data/exports/centrality_metrics.csv
         data/processed/factors_panel.csv
Outputs: data/exports/regression_coefficients.csv
         data/exports/regression_model_comparison.csv
         data/exports/vif_analysis.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED   = os.path.join(ROOT, "data", "processed")
EXPORTS_DIR = os.path.join(ROOT, "data", "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

SNAPSHOT_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & MERGE DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_regression_data() -> pd.DataFrame:
    """
    Load centrality metrics and factors panel; merge on (iso3, year).
    Apply log transformations and return analysis-ready DataFrame.
    """
    centrality_path = os.path.join(EXPORTS_DIR, "centrality_metrics.csv")
    factors_path    = os.path.join(PROCESSED,   "factors_panel.csv")

    if not os.path.exists(centrality_path):
        raise FileNotFoundError(
            "centrality_metrics.csv not found. Run 02_network_construction_centrality.py first."
        )
    if not os.path.exists(factors_path):
        raise FileNotFoundError(
            "factors_panel.csv not found. Run 01_data_collection_cleaning.py first."
        )

    print("Loading centrality metrics...")
    df_c = pd.read_csv(centrality_path)

    print("Loading factors panel...")
    df_f = pd.read_csv(factors_path)

    # Merge: centrality metrics provide the dependent variables
    # factors panel provides the independent variables (country-year level)
    merged = df_c.merge(df_f, on=["iso3", "year"], how="left")

    print(f"  Merged shape: {merged.shape}")
    print(f"  Columns: {merged.columns.tolist()}")

    # ── Log transformations ───────────────────────────────────────────────
    # GDP and population are highly skewed; log-transform for linearity
    for col in ["gdp_per_capita", "population"]:
        if col in merged.columns:
            merged[f"log_{col}"] = np.where(
                merged[col] > 0, np.log(merged[col]), np.nan
            )
        else:
            print(f"  WARNING: '{col}' not found in factors panel, will be NaN")
            merged[f"log_{col}"] = np.nan

    # Similarly log-transform in/out strength (heavy-tailed)
    for col in ["in_strength", "out_strength"]:
        if col in merged.columns:
            merged[f"log_{col}"] = np.where(
                merged[col] > 0, np.log(merged[col]), np.nan
            )

    # Fill conflict intensity NaN → 0
    if "conflict_intensity" in merged.columns:
        merged["conflict_intensity"] = merged["conflict_intensity"].fillna(0)
    else:
        merged["conflict_intensity"] = 0.0

    # Optional features — set to NaN if missing
    for col in ["unemployment", "education_index", "visa_openness_index", "climate_vulnerability"]:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 2. OLS REGRESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fit_ols_model(
    df:       pd.DataFrame,
    y_col:    str,
    x_cols:   list,
    model_name: str,
) -> tuple[object, pd.DataFrame]:
    """
    Fit an OLS model with specified X and Y columns.
    Drops rows with NaN in any required column.
    Returns (fitted model, coefficient DataFrame).
    """
    required = [y_col] + x_cols
    df_model = df[required].dropna()

    if len(df_model) < 30:
        print(f"  WARNING: Only {len(df_model)} complete cases for '{model_name}'. Skipping.")
        return None, pd.DataFrame()

    X = sm.add_constant(df_model[x_cols])
    y = df_model[y_col]

    model  = sm.OLS(y, X).fit()
    conf   = model.conf_int(alpha=0.05)
    conf.columns = ["ci_lower_95", "ci_upper_95"]

    coef_df = pd.DataFrame({
        "model":       model_name,
        "dependent":   y_col,
        "predictor":   model.params.index,
        "coefficient": model.params.values,
        "std_error":   model.bse.values,
        "t_stat":      model.tvalues.values,
        "p_value":     model.pvalues.values,
    }).join(conf.reset_index(drop=True))

    coef_df["n_obs"]      = model.nobs
    coef_df["r_squared"]  = model.rsquared
    coef_df["adj_r2"]     = model.rsquared_adj
    coef_df["aic"]        = model.aic
    coef_df["bic"]        = model.bic
    coef_df["significant_95"] = coef_df["p_value"] < 0.05

    return model, coef_df


def fit_model_pair(
    df: pd.DataFrame,
    y_col: str,
    direction: str,
) -> tuple[list, list]:
    """
    Fit baseline (3-factor) and full (7-factor) OLS models for a given
    dependent variable.
    Returns (list of coef DataFrames, list of summary dicts).
    """
    # Baseline predictors
    baseline_preds = [c for c in ["log_population", "log_gdp_per_capita", "conflict_intensity"]
                      if c in df.columns]
    # Full predictor set
    full_preds = baseline_preds + [
        c for c in ["unemployment", "education_index", "visa_openness_index", "climate_vulnerability"]
        if c in df.columns
    ]

    coef_dfs = []
    summary_rows = []

    for model_label, preds in [
        (f"baseline_{direction}", baseline_preds),
        (f"full_{direction}",     full_preds),
    ]:
        print(f"\n  Fitting {model_label} (y={y_col}, X={preds})")
        model, coef_df = fit_ols_model(df, y_col, preds, model_label)
        if model is None:
            continue
        coef_dfs.append(coef_df)
        summary_rows.append({
            "model":        model_label,
            "dependent":    y_col,
            "n_predictors": len(preds),
            "n_obs":        int(model.nobs),
            "r_squared":    model.rsquared,
            "adj_r2":       model.rsquared_adj,
            "aic":          model.aic,
            "bic":          model.bic,
            "f_stat":       model.fvalue,
            "f_p_value":    model.f_pvalue,
        })
        print(f"    R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}, "
              f"AIC = {model.aic:.1f}, n = {int(model.nobs)}")

    return coef_dfs, summary_rows


# ══════════════════════════════════════════════════════════════════════════════
# 3. VIF ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_vif(df: pd.DataFrame, x_cols: list, label: str) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for all predictors.
    VIF > 10 indicates problematic multicollinearity.
    """
    df_clean = df[x_cols].dropna()
    if len(df_clean) < len(x_cols) + 5:
        print(f"  WARNING: Insufficient data for VIF analysis in '{label}'")
        return pd.DataFrame()

    X = sm.add_constant(df_clean)
    vif_data = pd.DataFrame({
        "model":     label,
        "predictor": X.columns,
        "vif":       [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    vif_data["multicollinearity_flag"] = vif_data["vif"] > 10
    return vif_data


# ══════════════════════════════════════════════════════════════════════════════
# 4. YEAR-BY-YEAR COEFFICIENT EVOLUTION (for temporal analysis)
# ══════════════════════════════════════════════════════════════════════════════

def fit_year_by_year(df: pd.DataFrame, y_col: str, x_cols: list) -> pd.DataFrame:
    """
    Fit separate OLS models per snapshot year.
    Returns long-format DataFrame with (year, predictor, coefficient, p_value).
    """
    rows = []
    for year in sorted(df["year"].unique()):
        df_y = df[df["year"] == year]
        _, coef_df = fit_ols_model(df_y, y_col, x_cols, f"year_{year}")
        if coef_df.empty:
            continue
        coef_df["year"] = year
        rows.append(coef_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 4: REGRESSION ANALYSIS")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading and merging data...")
    df = load_regression_data()

    # Use log_in_strength and log_out_strength as continuous dependent variables
    # (more suitable for OLS than degree counts)
    dep_vars = {
        "inflows":  "log_in_strength",
        "outflows": "log_out_strength",
    }

    all_preds_full = [c for c in [
        "log_population", "log_gdp_per_capita", "conflict_intensity",
        "unemployment", "education_index", "visa_openness_index", "climate_vulnerability"
    ] if c in df.columns]

    # ── Pooled regressions (all years) ───────────────────────────────────────
    print("\n[2/5] Fitting pooled OLS models (baseline + full, inflows + outflows)...")
    all_coef_dfs   = []
    all_summary_rows = []

    for direction, y_col in dep_vars.items():
        if y_col not in df.columns:
            print(f"  Skipping {direction}: '{y_col}' not found")
            continue
        coef_list, summ_list = fit_model_pair(df, y_col, direction)
        all_coef_dfs.extend(coef_list)
        all_summary_rows.extend(summ_list)

    # ── VIF analysis ─────────────────────────────────────────────────────────
    print("\n[3/5] Computing VIF analysis...")
    vif_dfs = []
    available_preds = [c for c in all_preds_full if c in df.columns]
    if available_preds:
        vif_df_pooled = compute_vif(df, available_preds, "full_model_pooled")
        vif_dfs.append(vif_df_pooled)
        print("  VIF results (full model):")
        print(vif_df_pooled.to_string(index=False))

    # ── Year-by-year coefficient evolution ───────────────────────────────────
    print("\n[4/5] Fitting year-by-year models (for temporal coefficient evolution)...")
    # Only for inflows and full predictor set
    y_col_in = dep_vars.get("inflows", "log_in_strength")
    if y_col_in in df.columns and available_preds:
        ybyyear_df = fit_year_by_year(df, y_col_in, available_preds)
        if not ybyyear_df.empty:
            out_yby = os.path.join(EXPORTS_DIR, "regression_year_by_year.csv")
            ybyyear_df.to_csv(out_yby, index=False)
            print(f"  ✓ Saved: {out_yby}  ({len(ybyyear_df):,} rows)")

    # ── Export ───────────────────────────────────────────────────────────────
    print("\n[5/5] Exporting results...")

    if all_coef_dfs:
        coef_full = pd.concat(all_coef_dfs, ignore_index=True)
        out_coef  = os.path.join(EXPORTS_DIR, "regression_coefficients.csv")
        coef_full.to_csv(out_coef, index=False)
        print(f"  ✓ regression_coefficients.csv: {len(coef_full):,} rows")

    if all_summary_rows:
        summ_df  = pd.DataFrame(all_summary_rows)
        out_summ = os.path.join(EXPORTS_DIR, "regression_model_comparison.csv")
        summ_df.to_csv(out_summ, index=False)
        print(f"  ✓ regression_model_comparison.csv:")
        print(summ_df[["model", "n_obs", "r_squared", "adj_r2", "aic"]].to_string(index=False))

    if vif_dfs:
        vif_all  = pd.concat(vif_dfs, ignore_index=True)
        out_vif  = os.path.join(EXPORTS_DIR, "vif_analysis.csv")
        vif_all.to_csv(out_vif, index=False)
        print(f"  ✓ vif_analysis.csv: {len(vif_all):,} rows")

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
