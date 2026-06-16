import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score,
                              f1_score)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                      "axes.titlesize": 13, "axes.labelsize": 11})

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

INPUT_CSV = "data/raw/early_2012_2013_loan_sample_with_outcome.csv"
RARE_PURPOSE_THRESHOLD = 0.01


# Data Cleaning

# Helpers
def to_numeric_percent(s):
    s2 = s.astype(str).str.replace("%", "", regex=False).str.strip()
    s2 = s2.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s2, errors="coerce")

def clean_emp_length(s):
    s2 = s.astype(str).str.lower().str.strip()
    s2 = s2.replace({"": np.nan, "nan": np.nan, "none": np.nan, "n/a": np.nan})
    out = pd.Series(np.nan, index=s2.index, dtype="float64")
    out[s2.str.contains(r"<\s*1", na=False)] = 0.0
    out[s2.str.contains(r"10\+", na=False)] = 10.0
    digits = s2.str.extract(r"(\d+)")[0]
    out = out.fillna(pd.to_numeric(digits, errors="coerce"))
    return out

def parse_month_year(s):
    dt = pd.to_datetime(s, errors="coerce", format="%b-%Y")
    if dt.isna().mean() > 0.2:
        dt = pd.to_datetime(s, errors="coerce")
    return dt

def add_missing_flag_and_impute(df, col):
    df[f"{col}_missing"] = df[col].isna().astype(int)
    df[col] = df[col].fillna(df[col].median())

# Load
print(f"\nLoading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
print(f"Raw shape: {df.shape}")

# Step 1: Check for duplicates
n_dup = df.duplicated().sum()
n_dup_id = df["id"].duplicated().sum()
print(f"Duplicates: {n_dup} full-row, {n_dup_id} by id")
if n_dup > 0:
    df = df.drop_duplicates()
    print(f"  Removed {n_dup} duplicate rows")

print(f"Loan status:\n{df['loan_status'].value_counts().to_string()}")

# 2. Filter to loans with observed outcomes (exclude Current)
before = len(df)
non_current = ["Fully Paid", "Charged Off", "Default",
            "Late (31-120 days)", "Late (16-30 days)", "In Grace Period"]
df = df[df["loan_status"].isin(non_current)].copy()
print(f"\nNon-current loans: {before} -> {len(df)} (dropped {before-len(df)} Current with unknown outcomes)")

# 3. Clean target
df["loan_is_bad"] = df["loan_is_bad"].astype(int)

# Flag terminal vs non-terminal outcomes (for Q3 strict filtering)
terminal_statuses = ["Fully Paid", "Charged Off", "Default"]
df["is_terminal"] = df["loan_status"].isin(terminal_statuses).astype(int)
n_nonterminal = (df["is_terminal"] == 0).sum()
print(f"  Terminal outcomes: {df['is_terminal'].sum():,} | Non-terminal (Late/Grace): {n_nonterminal}")

# 4. Drop leakage
post_outcome = [
    "loan_status", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
]
post_decision = ["funded_amnt", "funded_amnt_inv"]
df.drop(columns=post_outcome + post_decision, inplace=True, errors="ignore")

# 5. Drop IDs, text, redundant, constant, sparse, high-missing
# Justification for each removal:
#   id, member_id           - Identifiers, no predictive value
#   emp_title, title, desc  - Free text, requires NLP (out of scope for prototype)
#   zip_code                - 799 levels, too sparse for meaningful signal
#   addr_state              - 46 levels, too sparse; also potential fairness concern
#   policy_code             - Constant (=1 for all rows)
#   collections_12_mths_ex_med - Near-constant (99.9% zeros)
#   application_type        - Constant (="INDIVIDUAL" for all rows)
#   pymnt_plan              - Only 3 "Y" cases; insufficient for statistical significance
#   initial_list_status     - Both categories show same default rate; funding
#                             mechanics independent of borrower creditworthiness
#   mths_since_last_record  - 95% missing
#   mths_since_last_major_derog - 86% missing

drop = ["id", "member_id", "emp_title", "title", "desc", "zip_code",
        "addr_state", "policy_code",
        "collections_12_mths_ex_med", "application_type",
        "pymnt_plan", "initial_list_status",
        "mths_since_last_record", "mths_since_last_major_derog"]
df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

# 6. Clean types
df["term"] = df["term"].astype(int)
df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")
df["revol_util"] = to_numeric_percent(df["revol_util"])
df["emp_length"] = clean_emp_length(df["emp_length"])

# 7. Date features
issue_dt = parse_month_year(df["issue_d"])
earliest_dt = parse_month_year(df["earliest_cr_line"])
months = (issue_dt.dt.year - earliest_dt.dt.year)*12 + (issue_dt.dt.month - earliest_dt.dt.month)
df["credit_history_years"] = (months / 12).where(months.notna(), np.nan)
df.drop(columns=["issue_d", "earliest_cr_line"], inplace=True, errors="ignore")

# 8. Imputation

# 8a. About account illogicality check
# Investigate combinations of {revol_bal, total_credit_rv, revol_util} for illogical rows.
# revol_util is the single source of truth for creditworthiness; we clean it,
# then drop revol_bal and total_credit_rv (whose predictive value is captured
# in revol_util, and total_credit_rv has ~30% missing).

if all(c in df.columns for c in ["revol_bal", "total_credit_rv", "revol_util"]):
    df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
    df["revol_bal"] = pd.to_numeric(df["revol_bal"], errors="coerce")
    df["total_credit_rv"] = pd.to_numeric(df["total_credit_rv"], errors="coerce")

    before_revol = len(df)

    # Scenario mapping:
    # Drop illogical rows where revolving data is contradictory/uninterpretable
    rb = df["revol_bal"]; tcr = df["total_credit_rv"]; ru = df["revol_util"]

    # Scenario 3: revol_bal>0, total_credit_rv=0 (impossible: balance without credit line)
    drop_s3 = (rb > 0) & (tcr == 0)
    # Scenario 4: revol_bal>0, total_credit_rv=NaN, revol_util is NaN or 0
    drop_s4 = (rb > 0) & tcr.isna() & (ru.isna() | (ru == 0))
    # Scenario 6: revol_bal=0, total_credit_rv=0 (no revolving history — ambiguous)
    drop_s6 = (rb == 0) & (tcr == 0)
    # Scenario 8: revol_bal=NaN, total_credit_rv=0 (no history vs perfect — ambiguous)
    drop_s8 = rb.isna() & (tcr == 0)
    # Scenario 10: revol_bal=NaN, total_credit_rv>0, revol_util=NaN
    drop_s10 = rb.isna() & (tcr > 0) & ru.isna()
    # Scenario 12: all three NaN
    drop_s12 = rb.isna() & tcr.isna() & ru.isna()

    illogical_revol = drop_s3 | drop_s4 | drop_s6 | drop_s8 | drop_s10 | drop_s12
    n_illogical = illogical_revol.sum()
    df = df[~illogical_revol].copy()

    print(f"\nRevolving account illogicality: dropped {n_illogical} rows "
          f"({before_revol} -> {len(df)})")

    # Now recalculate/impute revol_util for remaining rows
    ru_missing = df["revol_util"].isna()
    n_missing_ru = ru_missing.sum()

    # revol_bal>0, total_credit_rv>0, revol_util NaN/0 -> recalculate
    can_recalc = ru_missing & (df["revol_bal"] > 0) & (df["total_credit_rv"] > 0)
    df.loc[can_recalc, "revol_util"] = (
        df.loc[can_recalc, "revol_bal"] / df.loc[can_recalc, "total_credit_rv"] * 100
    ).clip(upper=100)

    # revol_bal=0 -> utilisation is 0% regardless
    can_zero = ru_missing & (df["revol_bal"] == 0)
    df.loc[can_zero, "revol_util"] = 0.0

    still_missing = df["revol_util"].isna().sum()
    print(f"revol_util imputation: {n_missing_ru} missing -> "
          f"{can_recalc.sum()} recalculated, "
          f"{can_zero.sum()} set to 0, "
          f"{still_missing} remaining -> median")
    df["revol_util_missing"] = df["revol_util"].isna().astype(int)
    df["revol_util"] = df["revol_util"].fillna(df["revol_util"].median())

    # Drop revol_bal and total_credit_rv: predictive value captured in revol_util,
    # and dropping resolves ~30% missingness in total_credit_rv without biased imputation

    df.drop(columns=["revol_bal", "total_credit_rv"], inplace=True, errors="ignore")
    print("Dropped revol_bal and total_credit_rv (captured in revol_util)")

# 8b. Other numeric imputations

# tot_coll_amt: impute with 0 (missing likely means zero collections, not "average")
if "tot_coll_amt" in df.columns:
    df["tot_coll_amt"] = pd.to_numeric(df["tot_coll_amt"], errors="coerce")
    df["tot_coll_amt_missing"] = df["tot_coll_amt"].isna().astype(int)
    n_miss_tc = df["tot_coll_amt"].isna().sum()
    df["tot_coll_amt"] = df["tot_coll_amt"].fillna(0)
    print(f"tot_coll_amt: {n_miss_tc} missing -> imputed with 0 (no collections)")

# Other imputations with missing flags (median)
for c in ["emp_length", "tot_cur_bal", "credit_history_years"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        add_missing_flag_and_impute(df, c)

# tot_cur_bal: create missing flag (borrowers who didn't report may be a specific risk group)
if "tot_cur_bal_missing" in df.columns:
    print(f"tot_cur_bal: missing flag created ({df['tot_cur_bal_missing'].sum()} missing -> median)")

if "mths_since_last_delinq" in df.columns:
    df["mths_since_last_delinq"] = pd.to_numeric(df["mths_since_last_delinq"], errors="coerce")
    df["has_delinq"] = df["mths_since_last_delinq"].notna().astype(int)
    df["mths_since_last_delinq"] = df["mths_since_last_delinq"].fillna(999)

# 9. Standardise categoricals
for c in ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.upper()

# Merge NONE into OTHER for home_ownership (42 NONE loans, too few for own category)
if "home_ownership" in df.columns:
    df.loc[df["home_ownership"] == "NONE", "home_ownership"] = "OTHER"
    print(f"home_ownership: merged NONE into OTHER -> {df['home_ownership'].value_counts().to_dict()}")

if "purpose" in df.columns:
    freq = df["purpose"].value_counts(normalize=True)
    rare = freq[freq < RARE_PURPOSE_THRESHOLD].index
    df.loc[df["purpose"].isin(rare), "purpose"] = "OTHER"

# 10. Data quality checks
# 10a. Validate: open_acc should not exceed total_acc
if "open_acc" in df.columns and "total_acc" in df.columns:
    bad_acc = (df["open_acc"] > df["total_acc"]).sum()
    print(f"Data quality: open_acc > total_acc = {bad_acc} rows (none expected)")

# 10b. Installment illogicality check
# installment is derivative of loan_amnt, int_rate, and term (amortization formula).
# Drop rows where the observed installment deviates >5% from the formula
# (1-5% discrepancy = rounding/fees; >5% = data error).
if all(c in df.columns for c in ["installment", "int_rate", "loan_amnt", "term"]):
    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    df["installment"] = pd.to_numeric(df["installment"], errors="coerce")
    r = (df["int_rate"] / 100) / 12
    n = df["term"]
    pv = df["loan_amnt"]
    calc_installment = (r * pv) / (1 - (1 + r)**(-n))
    pct_diff = ((df["installment"] - calc_installment) / calc_installment).abs()
    illogical_inst = pct_diff > 0.05
    n_illogical_inst = illogical_inst.sum()
    before_inst = len(df)
    df = df[~illogical_inst].copy()
    print(f"Installment illogicality: dropped {n_illogical_inst} rows with >5% "
          f"deviation from amortization formula ({before_inst} -> {len(df)})")

    # Drop installment: it is mathematically determined by loan_amnt, int_rate, term
    # and adds multicollinearity without new predictive information
    df.drop(columns=["installment"], inplace=True)
    print("Dropped installment (derivative of loan_amnt, int_rate, term)")

# 11. Force numeric
for c in ["loan_amnt", "annual_inc", "dti", "delinq_2yrs",
           "inq_last_6mths", "open_acc", "pub_rec", "acc_now_delinq"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 12. Drop total_acc (correlated with open_acc at ~0.66)
if "total_acc" in df.columns:
    df.drop(columns=["total_acc"], inplace=True)
    print("Dropped total_acc (correlated 0.66 with open_acc; open_acc is more informative)")

# 13. Derived features
grade_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
df["grade_num"] = df["grade"].map(grade_map)

# 14. Cap extreme values
# Cap annual_inc, tot_coll_amt, tot_cur_bal at 99th percentile to prevent
# distortion in models while keeping extreme observations (not dropping them)
for c in ["annual_inc", "tot_coll_amt", "tot_cur_bal"]:
    if c in df.columns:
        cap = df[c].quantile(0.99)
        n_capped = (df[c] > cap).sum()
        df[c] = df[c].clip(upper=cap)
        if n_capped > 0:
            print(f"Capped {c} at 99th percentile ({cap:,.0f}): {n_capped} values clipped")

# Cap discrete count variables at logical risk-based thresholds
# (derived from risk lift analysis: beyond these values, risk stabilises and counts drop)
count_caps = {
    "pub_rec": 3,         # Risk stabilises after 3 public records
    "delinq_2yrs": 3,     # High delinquency is rare; risk plateaus
    "inq_last_6mths": 6,  # >6 inquiries is extreme credit-seeking behaviour
    "open_acc": 35,        # ~99th percentile for account counts
}
for var, cap_val in count_caps.items():
    if var in df.columns:
        n_capped = (df[var] > cap_val).sum()
        df[var] = df[var].clip(upper=cap_val)
        if n_capped > 0:
            print(f"Capped {var} at logical limit {cap_val}: {n_capped} values clipped")

# 15. Catch-all imputation
for c in df.select_dtypes(include=[np.number]).columns:
    if df[c].isna().any(): df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes(include=["object"]).columns:
    if df[c].isna().any():
        m = df[c].mode()
        if len(m) > 0: df[c] = df[c].fillna(m[0])

# 16. Sanity filter
df = df[df["annual_inc"] > 0].copy()

# Save cleaned data
df.to_csv(f"{OUT}/ADA_FINAL_CLEANED.csv", index=False)
N = len(df)
print(f"\nCleaned: {df.shape} | Default rate: {df['loan_is_bad'].mean():.2%} | NaNs: {df.isna().sum().sum()}")
print(f"Saved: {OUT}/ADA_FINAL_CLEANED.csv")

# Part 2 (Q1): Simulated A/B Policy Comparison

print("\n\n" + "=" * 65)
print("  Q1: SIMULATED A/B POLICY COMPARISON")
print("=" * 65)

# Q1a: Policy Definitions

# Sub-grade default rate analysis reveals a significant risk cliff between
# C2 (~17% DR) and C3 (~22% DR). Policies are defined using sub-grades for
# finer granularity than grade-level cutoffs.
subgrade_order = sorted(df["sub_grade"].unique())
print("\nDefault Rate by Sub-Grade:")
for sg in subgrade_order:
    sub = df[df["sub_grade"]==sg]
    print(f"  {sg}: n={len(sub):>4,}, DR={sub['loan_is_bad'].mean():.1%}, "
          f"int={sub['int_rate'].mean():.1f}%")

# Policy A (Conservative): Approve sub-grades A1-C2
# Rationale: C2 is the last sub-grade before the risk cliff; beyond C2,
# default rates jump significantly without proportional interest compensation.
policy_a_subs = subgrade_order[:12]  # A1 through C2
df["policy_a"] = df["sub_grade"].isin(policy_a_subs).astype(int)

# Policy B (Selective Expansion): Approve A1-C2 unconditionally,
# plus C3-C5 borrowers with DTI < 20%
# Rationale: Rather than blanket approval of riskier sub-grades, Policy B
# uses a secondary behavioural filter (DTI) to cherry-pick lower-risk
# borrowers from the C3-C5 segment. DTI < 20% selects borrowers whose
# existing debt burden is manageable relative to income.
tier_1_subs = subgrade_order[:12]   # A1 through C2
tier_2_subs = subgrade_order[12:15] # C3 through C5
df["policy_b"] = (
    df["sub_grade"].isin(tier_1_subs) |
    (df["sub_grade"].isin(tier_2_subs) & (df["dti"] < 20))
).astype(int)

n_a = df["policy_a"].sum()
n_b = df["policy_b"].sum()
# Marginal loans: approved by B but not A (i.e., the conditional C3-C5 cohort)
marginal_mask = (df["policy_b"]==1) & (df["policy_a"]==0)
n_marg = marginal_mask.sum()

print(f"""
Policy A (Conservative): Approve sub-grades A1-C2
  Approvals: {n_a:,} / {N:,} ({n_a/N:.1%})

Policy B (Selective Expansion): A1-C2 + C3-C5 where DTI < 20%
  Approvals: {n_b:,} / {N:,} ({n_b/N:.1%})
  Marginal loans (C3-C5, DTI<20%): {n_marg:,}

  Key design choice: Policy B is NOT simply "approve more grades."
  It uses a secondary filter (DTI) to selectively expand into riskier
  sub-grades, testing whether behavioural screening can unlock volume
  without proportionally increasing default risk.
""")

# Q1b: Evaluation Metric & Hypotheses
app_a = df[df["policy_a"] == 1]
app_b = df[df["policy_b"] == 1]
d_a = app_a["loan_is_bad"].sum()
d_b = app_b["loan_is_bad"].sum()
dr_a = d_a / n_a
dr_b = d_b / n_b

marginal_loans = df[marginal_mask]
d_marg = marginal_loans["loan_is_bad"].sum()
dr_marg = d_marg / n_marg if n_marg > 0 else 0

print(f"""
Metric: Default Rate among approved loans = defaults / approved
  Policy A pool (A1-C2):        {d_a:,}/{n_a:,} = {dr_a:.2%}
  Marginal pool (C3-C5,DTI<20): {d_marg:,}/{n_marg:,} = {dr_marg:.2%}
  Policy B combined:            {d_b:,}/{n_b:,} = {dr_b:.2%}

  Rationale: Default rate is the most direct measure of loan book quality.
  It captures the proportion of approved loans that fail, which maps
  directly to the business objective of minimising losses from defaults.

Hypotheses (one-sided):
  H0: DR(B) <= DR(A)   (selective expansion does not increase risk)
  H1: DR(B) >  DR(A)   (selective expansion increases default rate)
  alpha = 0.05
""")

# Q1c: Policy Comparison

# Deterministic decomposition
diff = dr_b - dr_a
print(f"""
Deterministic Decomposition:
  Policy A (A1-C2) DR:             {dr_a:.2%}
  Marginal (C3-C5, DTI<20%) DR:   {dr_marg:.2%}
  Policy B combined DR:            {dr_b:.2%}
  Difference (B-A):                +{diff:.4f} ({diff*100:.2f} pp)
""")

# Bootstrap test (2,000 resamples)
np.random.seed(42)
boot_diffs = []
for _ in range(2000):
    sample = df.sample(n=N, replace=True)
    a = sample[sample["policy_a"]==1]
    b = sample[sample["policy_b"]==1]
    if len(a) > 0 and len(b) > 0:
        boot_diffs.append(b["loan_is_bad"].mean() - a["loan_is_bad"].mean())
boot_diffs = np.array(boot_diffs)
ci_lo = np.percentile(boot_diffs, 2.5)
ci_hi = np.percentile(boot_diffs, 97.5)
p_one = (boot_diffs <= 0).mean()
print(f"  Mean diff: +{boot_diffs.mean():.4f}")
print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  One-sided p: {p_one:.4f} -> {'REJECT H0' if p_one < 0.05 else 'FAIL TO REJECT'}")

# Summary table
rej_good_a = ((df["policy_a"]==0)&(df["loan_is_bad"]==0)).sum()
rej_good_b = ((df["policy_b"]==0)&(df["loan_is_bad"]==0)).sum()

print(f"\n{'Metric':<35} {'Policy A':>14} {'Policy B':>14} {'Diff':>10}")
for lbl, va, vb, vd in [
    ("Approved", f"{n_a:,}", f"{n_b:,}", f"+{n_b-n_a:,}"),
    ("Approval rate", f"{n_a/N:.1%}", f"{n_b/N:.1%}", f"+{(n_b-n_a)/N:.1%}"),
    ("Defaults", f"{d_a:,}", f"{d_b:,}", f"+{d_b-d_a:,}"),
    ("Default rate", f"{dr_a:.2%}", f"{dr_b:.2%}", f"+{diff:.2%}"),
    ("Avg interest", f"{app_a['int_rate'].mean():.2f}%", f"{app_b['int_rate'].mean():.2f}%", ""),
    ("Good rejected", f"{rej_good_a:,}", f"{rej_good_b:,}", f"-{rej_good_a-rej_good_b:,}"),
]:
    print(f"  {lbl:<33} {va:>14} {vb:>14} {vd:>10}")

# Financial impact using LGD model (industry-standard for unsecured loans)
LGD = 0.70  # Loss Given Default: 70% for unsecured personal loans (standard assumption)
print(f"\nFinancial Impact (LGD = {LGD:.0%} for unsecured loans):")
print("  Revenue = loan_amnt * (int_rate/100) for good loans (first-year interest proxy)")
print("  Loss = loan_amnt * LGD for defaulted loans")
for name, col in [("Policy A (A1-C2)", "policy_a"), ("Policy B (Selective)", "policy_b")]:
    app = df[df[col]==1]
    good = app[app["loan_is_bad"]==0]; bad = app[app["loan_is_bad"]==1]
    revenue = (good["loan_amnt"] * (good["int_rate"] / 100)).sum()
    losses = (bad["loan_amnt"] * LGD).sum()
    net = revenue - losses
    print(f"  {name}: Rev ${revenue:,.0f} - Loss ${losses:,.0f} = "
          f"Net ${net:,.0f} (${net/len(app):.0f}/loan)")

# Default Rate by Grade and Sub-Grade
print("\nDefault Rate by Grade:")
for g in ["A","B","C","D","E","F","G"]:
    sub = df[df["grade"]==g]
    if len(sub) > 0:
        print(f"  {g}: n={len(sub):,}, DR={sub['loan_is_bad'].mean():.2%}, int={sub['int_rate'].mean():.1f}%")

# Q1 Figures
# Figure 1: Default rate by sub-grade with policy boundaries
fig, ax = plt.subplots(figsize=(14, 6))
sg_dr = [df[df["sub_grade"]==sg]["loan_is_bad"].mean() for sg in subgrade_order]
sg_n = [len(df[df["sub_grade"]==sg]) for sg in subgrade_order]
colors_sg = []
for sg in subgrade_order:
    if sg in policy_a_subs: colors_sg.append("#27ae60")
    elif sg in tier_2_subs: colors_sg.append("#f39c12")
    else: colors_sg.append("#e74c3c")
bars = ax.bar(subgrade_order, sg_dr, color=colors_sg, edgecolor="white", lw=0.8, zorder=3)
for bar, n, d in zip(bars, sg_n, sg_dr):
    if n > 100:  # only annotate sub-grades with enough data
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{d:.0%}", ha="center", va="bottom", fontsize=7, rotation=45)
ax.axvline(11.5, color="#27ae60", ls="--", lw=2.5, label="Policy A cutoff (A1-C2)")
ax.axvline(14.5, color="#f39c12", ls="--", lw=2.5, label="Policy B expansion zone (C3-C5, DTI<20%)")
ax.set_xlabel("Sub-Grade"); ax.set_ylabel("Default Rate")
ax.set_title("Figure 1: Default Rate by Sub-Grade with Policy Boundaries")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(0, max(sg_dr)*1.25); ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_default_by_subgrade.png", bbox_inches="tight"); plt.close()

# Figure 2: Policy comparison summary
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
lbls = ["Policy A\n(A1-C2)", "Policy B\n(Selective)"]; cols2 = ["#27ae60", "#f39c12"]
ax = axes[0]; v = [n_a/N, n_b/N]
ax.bar(lbls, v, color=cols2, edgecolor="white"); ax.set_title("Approval Rate")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
for i,x in enumerate(v): ax.text(i, x+0.01, f"{x:.1%}", ha="center", fontweight="bold")
ax = axes[1]; v = [dr_a, dr_b]
ax.bar(lbls, v, color=cols2, edgecolor="white"); ax.set_title("Default Rate (Approved)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
for i,x in enumerate(v): ax.text(i, x+0.003, f"{x:.2%}", ha="center", fontweight="bold")
p_label = f"p={p_one:.4f}" if p_one > 0 else "p < 0.0005"
ax.annotate(f"Delta = +{diff:.2%}\nBootstrap {p_label}", xy=(0.5, max(v)*0.75),
            fontsize=10, ha="center", color="red", fontweight="bold")
ax = axes[2]; v = [rej_good_a, rej_good_b]
ax.bar(lbls, v, color=cols2, edgecolor="white"); ax.set_title("Good Loans Rejected\n(Opportunity Cost)")
for i,x in enumerate(v): ax.text(i, x+100, f"{x:,}", ha="center", fontweight="bold")
plt.suptitle("Figure 2: Policy A vs Policy B", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_policy_comparison.png", bbox_inches="tight"); plt.close()

# Q1d: Limitations
print("""
--- Q1d: Limitations ---
1. SELECTION BIAS: Only funded loans observed; rejected applicants are unobservable.
2. NON-INDEPENDENT POLICIES: Policy A is nested within Policy B (shared A1-C2 pool).
   Addressed via decomposition of shared vs marginal pools + bootstrap.
3. NO CAUSAL INFERENCE: Grade and sub-grade are endogenous (assigned by the lender
   based on borrower characteristics), so we cannot claim causal effects of policy.
4. TEMPORAL CONFOUNDING: Data spans May 2012 - Feb 2013 only; results may not
   generalise to different macroeconomic conditions.
5. CURRENT LOAN EXCLUSION: ~6,621 Current loans removed (unknown outcomes).
6. SIMPLIFIED FINANCIALS: LGD-based model uses first-year interest as revenue
   proxy and assumes 70% loss-given-default; does not capture recoveries,
   early repayment, or multi-year interest accumulation.
7. DELINQUENCY PROXY: Late/Grace Period loans treated as defaults but may recover.
""")
print("Q1 COMPLETE.\n")

# Part3 (Q2): Borrower Segmentation

# Q2a: Methodology
print("\n--- Q2a: Feature Selection & Methodology ---")
clust_feats = ["annual_inc", "dti", "loan_amnt", "int_rate", "revol_util",
               "credit_history_years", "delinq_2yrs", "inq_last_6mths",
               "open_acc", "pub_rec"]
print(f"Features ({len(clust_feats)}): {clust_feats}")

# Cap outliers for clustering input only (annual_inc already capped at 99th in cleaning;
# apply 99.5th here for clustering-specific tighter control)
X_clust = df[clust_feats].copy()
for c in ["annual_inc", "loan_amnt"]:
    X_clust[c] = X_clust[c].clip(upper=X_clust[c].quantile(0.995))

scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(X_clust)

# k selection
print("\nk selection:")
ks = range(2, 9); inertias = []; sils = []
for k in ks:
    km_test = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_test = km_test.fit_predict(X_clust_scaled)
    inertias.append(km_test.inertia_)
    # sample_size=10000 for computational efficiency (standard approximation for large N)
    s = silhouette_score(X_clust_scaled, labels_test, sample_size=10000, random_state=42)
    sils.append(s)
    print(f"  k={k}: Inertia={km_test.inertia_:>10,.0f}  Silhouette={s:.4f}")

# Figure 3: Elbow + Silhouette
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.plot(list(ks), inertias, "o-", color="#2c3e50", lw=2, ms=8)
a1.set_xlabel("k"); a1.set_ylabel("Inertia"); a1.set_title("Elbow Method"); a1.set_xticks(list(ks))
a2.plot(list(ks), sils, "o-", color="#c0392b", lw=2, ms=8)
a2.set_xlabel("k"); a2.set_ylabel("Silhouette"); a2.set_title("Silhouette Analysis"); a2.set_xticks(list(ks))
plt.suptitle("Figure 3: Optimal Number of Clusters", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_elbow_silhouette.png", bbox_inches="tight"); plt.close()

K = 4
print(f"\nSelected k={K}. Silhouette ~0.12 is typical for overlapping credit data.")
km_final = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_clust_scaled)

# Q2b: Cluster Profiles
print("\n--- Q2b: Cluster Profiles ---")
print("NOTE: loan_is_bad was NOT used in clustering. Labels are post-hoc.\n")

profile = df.groupby("cluster").agg(
    n=("loan_is_bad","size"), dr=("loan_is_bad","mean"),
    inc=("annual_inc","mean"), dti=("dti","mean"),
    loan=("loan_amnt","mean"), rate=("int_rate","mean"),
    util=("revol_util","mean"), hist=("credit_history_years","mean"),
    delinq=("delinq_2yrs","mean"), inq=("inq_last_6mths","mean"),
    pub_rec=("pub_rec","mean"),
).round(3)
profile_sorted = profile.sort_values("dr")

labels = {}
for rank, (c, r) in enumerate(profile_sorted.iterrows()):
    if rank == 0: labels[c] = "Low-Risk Prime"
    elif rank == 1: labels[c] = "Mid-Risk Mainstream"
    elif rank == 2: labels[c] = "High-Income Leveraged"
    else: labels[c] = "High-Risk Stressed"
df["cluster_label"] = df["cluster"].map(labels)

for c, r in profile_sorted.iterrows():
    print(f"  C{c} - {labels[c]} (n={r['n']:,.0f}, DR={r['dr']:.1%})")
    print(f"    Inc=${r['inc']:,.0f} DTI={r['dti']:.1f} Loan=${r['loan']:,.0f} "
          f"Rate={r['rate']:.1f}% Util={r['util']:.1f}% Hist={r['hist']:.1f}yr "
          f"Delinq={r['delinq']:.2f} Inq={r['inq']:.2f}")

# Figure 4: Cluster profiles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cm = df.groupby("cluster")[clust_feats].mean()
cm_z = (cm - cm.mean()) / cm.std()
cm_z.index = [f"C{i}: {labels[i]}" for i in cm_z.index]
sns.heatmap(cm_z.T, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0, ax=axes[0,0], linewidths=0.5)
axes[0,0].set_title("Cluster Feature Profiles (Standardised)")

dr_s = profile_sorted["dr"]
bc = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, K))
bl = [f"C{i}\n{labels[i]}" for i in dr_s.index]
bs = axes[0,1].bar(bl, dr_s.values, color=bc, edgecolor="white")
for bx, v in zip(bs, dr_s.values): axes[0,1].text(bx.get_x()+bx.get_width()/2, v+0.005, f"{v:.1%}", ha="center", fontweight="bold")
axes[0,1].set_ylabel("Default Rate"); axes[0,1].set_title("Default Rate by Cluster")
axes[0,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

sz = df["cluster"].value_counts().sort_index()
axes[1,0].bar([f"C{i}\n{labels[i]}" for i in sz.index], sz.values,
              color=plt.cm.Set2(np.linspace(0, 0.8, K)), edgecolor="white")
for i, v in enumerate(sz.values): axes[1,0].text(i, v+200, f"{v:,}", ha="center", fontweight="bold")
axes[1,0].set_ylabel("Loans"); axes[1,0].set_title("Cluster Sizes")

samp = df.sample(5000, random_state=42)
for ci in sorted(df["cluster"].unique()):
    sub = samp[samp["cluster"]==ci]
    axes[1,1].scatter(sub["annual_inc"], sub["dti"], alpha=0.35, s=12, label=f"C{ci}: {labels[ci]}")
axes[1,1].set_xlabel("Annual Income ($)"); axes[1,1].set_ylabel("DTI"); axes[1,1].set_xlim(0, 250000)
axes[1,1].set_title("Income vs DTI by Segment"); axes[1,1].legend(fontsize=8)
plt.suptitle("Figure 4: Borrower Segmentation", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_cluster_profiles.png", bbox_inches="tight"); plt.close()

# Q2c: Policy Performance by Cluster
print("\n--- Q2c: Policy Performance by Cluster ---\n")
print(f"{'Cluster':<30} {'n':>6} {'DR':>6} {'A DR':>6} {'B DR':>6} {'Marg n':>7} {'Marg DR':>8}")
print("-" * 75)
for ci in profile_sorted.index:
    sub = df[df["cluster"]==ci]
    pa = sub[sub["policy_a"]==1]; pb = sub[sub["policy_b"]==1]
    marg = sub[(sub["policy_b"]==1)&(sub["policy_a"]==0)]
    lbl = f"C{ci} ({labels[ci]})"
    adr = pa["loan_is_bad"].mean() if len(pa)>0 else np.nan
    bdr = pb["loan_is_bad"].mean() if len(pb)>0 else np.nan
    mdr = marg["loan_is_bad"].mean() if len(marg)>0 else np.nan
    mdr_str = f"{mdr:>7.1%}" if not np.isnan(mdr) else "    N/A"
    print(f"  {lbl:<28} {len(sub):>6,} {sub['loan_is_bad'].mean():>5.1%} {adr:>5.2%} {bdr:>5.2%} {len(marg):>7,} {mdr_str}")

# Figure 5: Policy by cluster
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
cs = list(profile_sorted.index); x = np.arange(len(cs)); w = 0.35
bl_cs = [f"C{c}\n{labels[c]}" for c in cs]
ar_a = [df[df["cluster"]==c]["policy_a"].mean() for c in cs]
ar_b = [df[df["cluster"]==c]["policy_b"].mean() for c in cs]
axes[0].bar(x-w/2, ar_a, w, label="Policy A (A1-C2)", color="#27ae60", edgecolor="white")
axes[0].bar(x+w/2, ar_b, w, label="Policy B (Selective)", color="#f39c12", edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(bl_cs, fontsize=9)
axes[0].set_ylabel("Approval Rate"); axes[0].set_title("Approval Rate by Cluster")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); axes[0].legend()
dr_a_l = [df[(df["cluster"]==c)&(df["policy_a"]==1)]["loan_is_bad"].mean() for c in cs]
dr_b_l = [df[(df["cluster"]==c)&(df["policy_b"]==1)]["loan_is_bad"].mean() for c in cs]
axes[1].bar(x-w/2, dr_a_l, w, label="Policy A (A1-C2)", color="#27ae60", edgecolor="white")
axes[1].bar(x+w/2, dr_b_l, w, label="Policy B (Selective)", color="#f39c12", edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(bl_cs, fontsize=9)
axes[1].set_ylabel("Default Rate"); axes[1].set_title("Default Rate by Cluster")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); axes[1].legend()
plt.suptitle("Figure 5: Policy Performance Across Borrower Segments", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_policy_by_cluster.png", bbox_inches="tight"); plt.close()

# Save dataset with clusters
df.to_csv(f"{OUT}/ADA_FINAL_WITH_CLUSTERS.csv", index=False)
print("\nQ2 COMPLETE.\n")

# Part4 (Q3): Deep Learning For Default Prediction


# Note: Clustering above was fit on full dataset for Q2 segmentation analysis.
# For Q3 evaluation, we refit clusters on training data only (see below).

# Q3a: Prediction Task and Data Split
print("\n--- Q3a: Prediction Task and Data Split ---")

# For Q3, restrict to terminal outcomes only (Fully Paid / Charged Off / Default).
# The 505 Late/Grace Period loans have ambiguous labels and are excluded from
# model training and evaluation. Q1/Q2 used the full 43,379-loan dataset.
df_q3 = df[df["is_terminal"] == 1].copy()
print(f"Q3 dataset: {len(df_q3):,} loans (terminal outcomes only)")
print(f"  Excluded: {len(df) - len(df_q3)} Late/Grace Period loans with ambiguous labels")
print(f"  Default rate: {df_q3['loan_is_bad'].mean():.2%}")

y = df_q3["loan_is_bad"].values
exclude_cols = ["loan_is_bad", "cluster", "cluster_label", "grade_num",
                "policy_a", "policy_b", "is_terminal"]
# Note: grade is excluded because sub_grade fully determines grade (e.g., A1 -> A).
# Including both would create redundant correlated one-hot features.
# grade_num is also excluded as it's a derived feature used only for Q1/Q2.
cat_cols = ["sub_grade", "home_ownership", "verification_status", "purpose"]
num_cols = [c for c in df_q3.columns if c not in exclude_cols and c not in cat_cols
            and df_q3[c].dtype in ["int64", "float64"]]

print(f"Numeric features: {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

# SPLIT FIRST, then fit preprocessing on training data only (no leakage)
N_q3 = len(df_q3)
idx_all = df_q3.index.values
idx_temp, idx_test, y_temp, y_test = train_test_split(
    idx_all, y, test_size=0.15, random_state=42, stratify=y)
idx_train, idx_val, y_train, y_val = train_test_split(
    idx_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

# Fit encoder and scaler on TRAINING data only
try:
    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
except TypeError:
    ohe = OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore")
ohe.fit(df_q3.loc[idx_train, cat_cols])

scaler_nn = StandardScaler()
scaler_nn.fit(df_q3.loc[idx_train, num_cols])

# Transform each split using training-fitted transformers
def build_X(indices):
    X_num = scaler_nn.transform(df_q3.loc[indices, num_cols])
    X_cat = ohe.transform(df_q3.loc[indices, cat_cols])
    return np.hstack([X_num, X_cat])

X_train = build_X(idx_train)
X_val   = build_X(idx_val)
X_test  = build_X(idx_test)
n_features = X_train.shape[1]
print(f"Total features: {n_features}")

# Fit clusters on TRAINING data only for clean Q3 evaluation
km_train = KMeans(n_clusters=K, random_state=42, n_init=10)
clust_train_df = df_q3.loc[idx_train, clust_feats].copy()
for c in ["annual_inc", "loan_amnt"]:
    clust_train_df[c] = clust_train_df[c].clip(upper=clust_train_df[c].quantile(0.995))
scaler_clust_q3 = StandardScaler()
X_clust_train = scaler_clust_q3.fit_transform(clust_train_df)
km_train.fit(X_clust_train)

# Predict clusters for test set using training-fitted model
clust_test_df = df_q3.loc[idx_test, clust_feats].copy()
for c in ["annual_inc", "loan_amnt"]:
    clust_test_df[c] = clust_test_df[c].clip(upper=clust_train_df[c].quantile(0.995))
X_clust_test = scaler_clust_q3.transform(clust_test_df)
test_clusters = km_train.predict(X_clust_test)
# Map to labels using training cluster profiles
train_profile = pd.DataFrame({"cluster": km_train.predict(X_clust_train), "bad": y_train})
train_dr = train_profile.groupby("cluster")["bad"].mean().sort_values()
q3_labels = {}
for rank, (c, _) in enumerate(train_dr.items()):
    if rank == 0: q3_labels[c] = "Low-Risk Prime"
    elif rank == 1: q3_labels[c] = "Mid-Risk Mainstream"
    elif rank == 2: q3_labels[c] = "High-Income Leveraged"
    else: q3_labels[c] = "High-Risk Stressed"
print("Preprocessing fit on training data only (no leakage).")
print("Clusters refit on training data for clean evaluation.")

print(f"\n  Train:      {len(y_train):>6,} ({len(y_train)/N_q3:.1%})  DR: {y_train.mean():.2%}")
print(f"  Validation: {len(y_val):>6,} ({len(y_val)/N_q3:.1%})  DR: {y_val.mean():.2%}")
print(f"  Test:       {len(y_test):>6,} ({len(y_test)/N_q3:.1%})  DR: {y_test.mean():.2%}")

# Q3b: Model Design and Training
print("\n--- Q3b: Model Design and Training ---")

n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
class_weight = {0: 1.0, 1: n_neg / n_pos}
print(f"Class weights: {{0: 1.0, 1: {class_weight[1]:.2f}}}")

model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy", metrics=["AUC"])

print("\nArchitecture: Input -> 128 -> BN -> Drop(0.3) -> 64 -> BN -> Drop(0.3) -> 32 -> Drop(0.2) -> 1(sigmoid)")
print(f"Parameters: {model.count_params():,}")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_AUC", patience=10, mode="max", restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_AUC", factor=0.5, patience=5, mode="max", min_lr=1e-6)

print("Training...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=256, class_weight=class_weight,
                    callbacks=[early_stop, reduce_lr], verbose=0)

best_epoch = np.argmax(history.history["val_AUC"]) + 1
print(f"Best epoch: {best_epoch} (val AUC: {max(history.history['val_AUC']):.4f})")

# Figure 6: Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(history.history["loss"], label="Train"); ax1.plot(history.history["val_loss"], label="Val")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(history.history["AUC"], label="Train"); ax2.plot(history.history["val_AUC"], label="Val")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC"); ax2.set_title("AUC"); ax2.legend(); ax2.grid(alpha=0.3)
plt.suptitle("Figure 6: Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_training_curves.png", bbox_inches="tight"); plt.close()

# Q3c: Model Evaluation
print("\n--- Q3c: Model Evaluation ---")

y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

auc_val = roc_auc_score(y_test, y_prob)
ap_val = average_precision_score(y_test, y_prob)
f1_val = f1_score(y_test, y_pred)

print(f"\nTest Set:")
print(f"  ROC-AUC:           {auc_val:.4f}  (~{auc_val*100:.0f}% of pairs ranked correctly)")
print(f"  Avg Precision:     {ap_val:.4f}  ({ap_val/y_test.mean():.1f}x improvement over baseline)")
print(f"  F1 (@ 0.5):        {f1_val:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Good','Default'], digits=3)}")

# Figure 7: ROC + PR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax1.plot(fpr, tpr, lw=2, color="#2c3e50", label=f"Model (AUC={auc_val:.3f})")
ax1.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC Curve"); ax1.legend(); ax1.grid(alpha=0.3)
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ax2.plot(rec, prec, lw=2, color="#c0392b", label=f"Model (AP={ap_val:.3f})")
ax2.axhline(y_test.mean(), color="k", ls="--", lw=1, alpha=0.5, label=f"Baseline ({y_test.mean():.3f})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall"); ax2.legend(); ax2.grid(alpha=0.3)
plt.suptitle("Figure 7: Model Discrimination", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_roc_pr_curves.png", bbox_inches="tight"); plt.close()

# Performance by cluster (using training-fitted clusters)
print("\nPerformance by Cluster:")
print(f"  {'Cluster':<28} {'n':>5} {'DR':>6} {'AUC':>6} {'AP':>6}")
print("  " + "-" * 55)
cluster_results = []
for ci in sorted(q3_labels.keys()):
    mask = test_clusters == ci
    if mask.sum() < 20: continue
    auc_c = roc_auc_score(y_test[mask], y_prob[mask]) if len(np.unique(y_test[mask]))>1 else np.nan
    ap_c = average_precision_score(y_test[mask], y_prob[mask]) if len(np.unique(y_test[mask]))>1 else np.nan
    print(f"  C{ci} ({q3_labels[ci]:<22}) {mask.sum():>5} {y_test[mask].mean():>5.1%} {auc_c:>5.3f} {ap_c:>5.3f}")
    cluster_results.append({"cluster": ci, "label": q3_labels[ci], "n": mask.sum(),
                            "dr": y_test[mask].mean(), "auc": auc_c})

# Figure 8: AUC by cluster
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
cr = sorted(cluster_results, key=lambda x: x["dr"])
cl = [f"C{r['cluster']}\n{r['label']}" for r in cr]
bc = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(cr)))
bars = ax1.bar(cl, [r["auc"] for r in cr], color=bc, edgecolor="white")
for bx, r in zip(bars, cr): ax1.text(bx.get_x()+bx.get_width()/2, r["auc"]+0.005, f"{r['auc']:.3f}", ha="center", fontweight="bold", fontsize=9)
ax1.set_ylabel("ROC-AUC"); ax1.set_title("Model AUC by Cluster"); ax1.set_ylim(0.5, max(r["auc"] for r in cr)*1.1); ax1.grid(axis="y", alpha=0.3)
bars2 = ax2.bar(cl, [r["dr"] for r in cr], color=bc, edgecolor="white")
for bx, r in zip(bars2, cr): ax2.text(bx.get_x()+bx.get_width()/2, r["dr"]+0.005, f"{r['dr']:.1%}", ha="center", fontweight="bold", fontsize=9)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax2.set_ylabel("Default Rate"); ax2.set_title("Default Rate by Cluster (Test)"); ax2.grid(axis="y", alpha=0.3)
plt.suptitle("Figure 8: Model Performance Across Borrower Segments", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig8_cluster_performance.png", bbox_inches="tight"); plt.close()

# Model-based policy comparison
print("\nModel-Based Policy Comparison (test set):")
test_df = df_q3.loc[idx_test].copy()
test_df["prob_default"] = y_prob; test_df["actual"] = y_test
for name, thresh in [("Model (p<0.20)", 0.20), ("Model (p<0.25)", 0.25), ("Model (p<0.30)", 0.30)]:
    app = test_df[test_df["prob_default"] < thresh]
    if len(app)==0: continue
    print(f"  {name}: Approved {len(app):,} ({len(app)/len(test_df):.1%}), DR={app['actual'].mean():.2%}")

# Compare with rule-based policies from Q1
sg_order = sorted(df["sub_grade"].unique())
pol_a_subs = sg_order[:12]  # A1-C2
pol_b_tier2 = sg_order[12:15]  # C3-C5
app_a_test = test_df[test_df["sub_grade"].isin(pol_a_subs)]
app_b_test = test_df[
    test_df["sub_grade"].isin(pol_a_subs) |
    (test_df["sub_grade"].isin(pol_b_tier2) & (test_df["dti"] < 20))
]
print(f"  Policy A (A1-C2): Approved {len(app_a_test):,} ({len(app_a_test)/len(test_df):.1%}), DR={app_a_test['actual'].mean():.2%}")
print(f"  Policy B (Selective): Approved {len(app_b_test):,} ({len(app_b_test)/len(test_df):.1%}), DR={app_b_test['actual'].mean():.2%}")

model.save(f"{OUT}/q3_model.keras")
print("\nQ3 COMPLETE.")

print("\nALL PARTS COMPLETE")

print(f"Outputs in: {OUT}/")
print(f"  ADA_FINAL_CLEANED.csv, ADA_FINAL_WITH_CLUSTERS.csv")
print(f"  fig1-fig8 (.png)")
print(f"  q3_model.keras")
