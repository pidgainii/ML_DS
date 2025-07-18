import pandas as pd
from scipy import stats

try:
    dataframe = pd.read_csv('spotify-2023.csv', encoding='latin')
    
    print(dataframe.head())

    dataframe['released_year'] = dataframe['released_year'].astype(str)
    dataframe['released_month'] = dataframe['released_month'].astype(str)

    contingency_table = pd.crosstab(dataframe['released_year'], dataframe['released_month'])

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\nChi-Squared Test between 'released_year' and 'released_month':")
    print(f"Chi-Squared statistic: {chi2_stat}")
    print(f"P-value: {p_value}")
    print(f"Degrees of freedom: {dof}")

    if p_value > 0.05:
        print("There is no significant association between 'released_year' and 'released_month'.")
    else:
        print("There is a significant association between 'released_year' and 'released_month'.")

except Exception as e:
    print(f"An error occurred: {e}")
