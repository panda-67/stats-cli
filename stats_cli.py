import os
import sys
import itertools
import pandas as pd
import tkinter as tk
from scipy import stats
from tkinter import filedialog
from InquirerPy import inquirer
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# -------------------
# Utility
# -------------------
def load_file():
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename(
        title="Select an Excel file", filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if not file:
        print("No file selected. Exiting.")
        sys.exit(0)

    df = pd.read_excel(file)
    name = os.path.basename(file)
    rows, cols = df.shape
    print(f"\nLoaded {name} with {rows} rows and {cols} columns.")
    return df


def choose_columns(df):
    dep_var = inquirer.select(
        message="Choose dependent variable:",
        choices=df.columns.tolist(),
    ).execute()

    group_var = inquirer.select(
        message="Choose grouping variable:",
        choices=df.columns.tolist(),
    ).execute()
    return dep_var, group_var


# -------------------
# t-TESTS
# -------------------
def independent_ttest(df, dep_var, group_var):
    unique_groups = df[group_var].dropna().unique().tolist()

    if len(unique_groups) > 2:
        selected = inquirer.checkbox(
            message="Select exactly two groups for the t-test:",
            choices=unique_groups,
            validate=lambda r: len(r) == 2 or "Select exactly 2 groups",
        ).execute()
        df = df[df[group_var].isin(selected)]

    groups = [g[dep_var].dropna().to_numpy() for _, g in df.groupby(group_var)]

    if len(groups) != 2:
        print("Error: t-test requires exactly 2 groups.")
        return

    stat, p = stats.ttest_ind(groups[0], groups[1])
    print(f"\nIndependent samples t-test: t={stat:.4f}, p={p:.4f}")


def paired_ttest(df, dep_var, group_var):
    unique_groups = df[group_var].dropna().unique().tolist()

    if len(unique_groups) > 2:
        selected = inquirer.checkbox(
            message="Select exactly two groups for the t-test:",
            choices=unique_groups,
            validate=lambda r: len(r) == 2 or "Select exactly 2 groups",
        ).execute()
        df = df[df[group_var].isin(selected)]
        unique_groups = df[group_var].dropna().unique().tolist()

    if len(unique_groups) != 2:
        print("Error: Paired t-test requires exactly 2 conditions.")
        return

    id_var = inquirer.select(
        message="Choose subject identifier column:",
        choices=df.columns.tolist(),
    ).execute()

    wide = df.pivot(index=id_var, columns=group_var, values=dep_var)
    if wide.shape[1] != 2:
        print("Error: Paired t-test requires exactly 2 conditions")
        return

    g1, g2 = wide.iloc[:, 0].dropna(), wide.iloc[:, 1].dropna()
    if len(g1) != len(g2):
        print("Warning: unequal pairs, dropping mismatches.")
        min_len = min(len(g1), len(g2))
        g1, g2 = g1.iloc[:min_len], g2.iloc[:min_len]

    stat, p = stats.ttest_rel(g1, g2)
    print(f"\nPaired samples t-test: t={stat:.4f}, p={p:.4f}")


def one_sample_ttest(df, dep_var):
    test_val = inquirer.text(
        message="Enter the test mean value (e.g., 0):",
        validate=lambda v: v.replace(".", "", 1).isdigit() or "Must be a number",
    ).execute()
    test_val = float(test_val)

    values = df[dep_var].dropna().to_numpy()
    stat, p = stats.ttest_1samp(values, test_val)
    print(f"\nOne-sample t-test: t={stat:.4f}, p={p:.4f}")


# -------------------
# ANOVA + Posthoc
# -------------------
def bonferroni_posthoc(df, dep_var, group_var):
    groups_list = df[group_var].unique()
    pairs, pvals = [], []
    for g1, g2 in itertools.combinations(groups_list, 2):
        vals1 = df[df[group_var] == g1][dep_var].dropna()
        vals2 = df[df[group_var] == g2][dep_var].dropna()
        _, pval = stats.ttest_ind(vals1, vals2)
        pairs.append((g1, g2))
        pvals.append(pval)

    reject, pvals_corr, _, _ = multipletests(pvals, method="bonferroni")

    print("\nBonferroni post-hoc results:")
    for (g1, g2), raw_p, adj_p, sig in zip(pairs, pvals, pvals_corr, reject):
        print(f"{g1} vs {g2}: raw p={raw_p:.4f}, adj p={adj_p:.4f}, sig={sig}")


def run_anova(df, dep_var, group_var):
    groups = [g[dep_var].dropna().to_numpy() for _, g in df.groupby(group_var)]
    stat, p = stats.f_oneway(*groups)
    print(f"\nANOVA: F={stat:.4f}, p={p:.4f}")

    if p < 0.05:
        posthoc_choice = inquirer.select(
            message="Select post-hoc test:",
            choices=[
                "1 = Tukey HSD",
                "2 = Bonferroni (pairwise t-tests)",
            ],
        ).execute()
        posthoc_choice = posthoc_choice.split(" = ")[0]

        if posthoc_choice == "1":
            tukey = pairwise_tukeyhsd(df[dep_var], df[group_var])
            print("\nTukey HSD results:")
            print(tukey.summary())

        elif posthoc_choice == "2":
            bonferroni_posthoc(df, dep_var, group_var)


# -------------------
# MAIN
# -------------------


def main():
    df = load_file()
    dep_var, group_var = choose_columns(df)

    choice = inquirer.select(
        message="Select test:",
        choices=[
            "1 = t-test",
            "2 = One-way ANOVA",
        ],
    ).execute()
    choice = choice.split(" = ")[0]

    if choice == "1":
        t_choice = inquirer.select(
            message="Select type of t-test:",
            choices=[
                "1 = Independent samples t-test",
                "2 = Paired samples t-test",
                "3 = One-sample t-test",
            ],
        ).execute()
        t_choice = t_choice.split(" = ")[0]

        if t_choice == "1":
            independent_ttest(df, dep_var, group_var)
        elif t_choice == "2":
            paired_ttest(df, dep_var, group_var)
        elif t_choice == "3":
            one_sample_ttest(df, dep_var)

    elif choice == "2":
        run_anova(df, dep_var, group_var)
    else:
        print("Invalid choice.")

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
