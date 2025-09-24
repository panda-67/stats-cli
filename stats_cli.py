import os
import pandas as pd
import questionary
from scipy import stats
from InquirerPy import inquirer
import tkinter as tk
from tkinter import filedialog


def main():
    # Find all Excel files in current directory
    # files = glob.glob("*.xlsx")
    #
    # if not files:
    #     print("No Excel files found in this directory.")
    #     exit(1)

    # # Let user pick one
    # file = inquirer.select(message="Choose an Excel file:", choices=files).execute()

    # Hide root window
    root = tk.Tk()
    root.withdraw()

    file = filedialog.askopenfilename(
        title="Select an Excel file", filetypes=[("Excel files", "*.xlsx *.xls")]
    )

    if not file:
        print("No file selected. Exiting.")
        exit(1)

    df = pd.read_excel(file)

    print(
        f"\nLoaded {os.path.basename(file)} with {df.shape[0]} rows and {
            df.shape[1]
        } columns."
    )

    dep_var = questionary.select(
        "Choose dependent variable:", choices=df.columns.tolist()
    ).ask()

    group_var = questionary.select(
        "Choose grouping variable:", choices=df.columns.tolist()
    ).ask()

    # Choose test
    choice = questionary.select(
        "Select test:", choices=["1 = t-test (two groups)", "2 = One-way ANOVA"]
    ).ask()

    # Extract the numeric choice (since you prefixed with 1/2)
    choice = choice.split(" = ")[0]

    # Group data
    groups = [g[dep_var].dropna().values for _, g in df.groupby(group_var)]

    if choice == "1":
        unique_groups = df[group_var].unique().tolist()

        if len(unique_groups) > 2:
            selected = inquirer.checkbox(
                message="Select exactly two groups for the t-test:",
                choices=unique_groups,
                validate=lambda result: len(result) == 2
                or "Please select exactly 2 groups",
                transformer=lambda result: ", ".join(result),
            ).execute()

            if not selected or len(selected) != 2:
                print("Error: You must select exactly 2 groups.")
                return

            # Subset to only those groups
            df = df[df[group_var].isin(selected)]

        # Now collect groups properly
        groups = [g[dep_var].dropna().to_numpy() for _, g in df.groupby(group_var)]

        if len(groups) != 2:
            print("Error: t-test requires exactly 2 groups.")
            return

        stat, p = stats.ttest_ind(groups[0], groups[1])
        print(f"\nT-test results: t={stat:.4f}, p={p:.4f}")

    elif choice == "2":
        stat, p = stats.f_oneway(*groups)
        print(f"\nANOVA results: F={stat:.4f}, p={p:.4f}")

    else:
        print("Invalid choice.")

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
