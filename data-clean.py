import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_and_explore_data(filepath):
    df = pd.read_csv(filepath)
    return df


def define_variables(df):
    variable_mapping = {
        'Outcome Variable': [],
        'Treatment Variable (WFH)': [],
        'Control Variables - Demographics': [],
        'Control Variables - Economic': [],
        'Control Variables - Geographic': []
    }

    car_keywords = ['car', 'vehicle', 'auto', 'ownership', 'cars', 'vehicles']
    wfh_keywords = ['wfh', 'work_from_home', 'remote', 'telework', 'home']
    income_keywords = ['income', 'earnings', 'wage', 'salary']
    education_keywords = ['education', 'educ', 'degree', 'college', 'bachelor']
    occupation_keywords = ['occupation', 'occup', 'white_collar', 'professional']
    demographic_keywords = ['population', 'pop', 'age', 'gender', 'race', 'ethnic']

    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in car_keywords):
            variable_mapping['Outcome Variable'].append(col)
        elif any(kw in col_lower for kw in wfh_keywords):
            variable_mapping['Treatment Variable (WFH)'].append(col)
        elif any(kw in col_lower for kw in income_keywords):
            variable_mapping['Control Variables - Economic'].append(col)
        elif any(kw in col_lower for kw in education_keywords + occupation_keywords):
            variable_mapping['Control Variables - Demographics'].append(col)
        elif any(kw in col_lower for kw in demographic_keywords):
            variable_mapping['Control Variables - Demographics'].append(col)

    return variable_mapping


def clean_data(df):
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)

    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)

    if df_clean.duplicated().sum() > 0:
        df_clean.drop_duplicates(inplace=True)

    return df_clean


def descriptive_statistics(df_clean, variable_mapping):
    create_visualizations(df_clean, variable_mapping)
    return df_clean


def create_visualizations(df_clean, variable_mapping):
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        sample_cols = numeric_cols[:min(6, len(numeric_cols))]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for idx, col in enumerate(sample_cols):
            if idx < len(axes):
                ax = axes[idx]
                df_clean[col].hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                mean_val = df_clean[col].mean()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.legend()
        for idx in range(len(sample_cols), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        plt.savefig('figures/01_variable_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    if len(sample_cols) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for idx, col in enumerate(sample_cols):
            if idx < len(axes):
                ax = axes[idx]
                df_clean.boxplot(column=col, ax=ax, patch_artist=True,
                                 boxprops=dict(facecolor='lightblue', color='navy'))
                ax.set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
                ax.set_ylabel(col)
        for idx in range(len(sample_cols), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        plt.savefig('figures/02_box_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

    if len(numeric_cols) >= 5:
        corr_cols = numeric_cols[:min(15, len(numeric_cols))]
        corr_matrix = df_clean[corr_cols].corr()
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    missing = df_clean.isnull().sum()
    missing = missing[missing > 0]
    plt.figure(figsize=(12, 6))
    if len(missing) > 0:
        missing_sorted = missing.sort_values(ascending=False)
        missing_top = missing_sorted.head(20) if len(missing_sorted) > 20 else missing_sorted
        plt.barh(range(len(missing_top)), missing_top.values, color='coral', edgecolor='black')
        plt.yticks(range(len(missing_top)), missing_top.index, fontsize=9)
        plt.xlabel('Number of Missing Values', fontsize=11, fontweight='bold')
        plt.title('Top 20 Variables with Missing Values', fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, 'No Missing Values in Cleaned Dataset',
                 ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('figures/04_missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()

    outcome_vars = variable_mapping.get('Outcome Variable', [])
    treatment_vars = variable_mapping.get('Treatment Variable (WFH)', [])
    plt.figure(figsize=(10, 8))
    if outcome_vars and treatment_vars:
        outcome_var = outcome_vars[0]
        treatment_var = treatment_vars[0]
        if outcome_var in df_clean.columns and treatment_var in df_clean.columns:
            plt.scatter(df_clean[treatment_var], df_clean[outcome_var],
                        alpha=0.5, s=20, c='steelblue', edgecolors='navy', linewidth=0.5)
            z = np.polyfit(df_clean[treatment_var].dropna(), df_clean[outcome_var].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_clean[treatment_var].min(), df_clean[treatment_var].max(), 100)
            plt.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend line')
            plt.xlabel(treatment_var, fontsize=12, fontweight='bold')
            plt.ylabel(outcome_var, fontsize=12, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/05_key_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()


def export_summary(df_clean, variable_mapping):
    numeric_df = df_clean.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        desc_stats = numeric_df.describe()
        desc_stats.to_csv('output/descriptive_statistics.csv')

    var_doc = []
    for category, vars_list in variable_mapping.items():
        for var in vars_list:
            if var in df_clean.columns:
                var_doc.append({
                    'Category': category,
                    'Variable Name': var,
                    'Data Type': str(df_clean[var].dtype),
                    'Missing Values': df_clean[var].isnull().sum(),
                    'Mean': df_clean[var].mean() if df_clean[var].dtype in [np.float64, np.int64] else None,
                    'Std': df_clean[var].std() if df_clean[var].dtype in [np.float64, np.int64] else None,
                    'Min': df_clean[var].min() if df_clean[var].dtype in [np.float64, np.int64] else None,
                    'Max': df_clean[var].max() if df_clean[var].dtype in [np.float64, np.int64] else None
                })

    pd.DataFrame(var_doc).to_csv('output/variable_documentation.csv', index=False)
    df_clean.to_csv('output/cleaned-usa.csv', index=False)


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    filepath = 'data/usa_data.csv'
    df = load_and_explore_data(filepath)
    variable_mapping = define_variables(df)
    df_clean = clean_data(df)
    df_final = descriptive_statistics(df_clean, variable_mapping)
    export_summary(df_final, variable_mapping)