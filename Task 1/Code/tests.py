import pandas as pd
from scipy import stats

try:
    # Loads CSV file
    dataframe = pd.read_csv('spotify-2023.csv', encoding = 'latin')


    data1name = "in_apple_playlists";
    data2name = "in_spotify_playlists";


    data1 = dataframe[data1name].dropna()  # dropna() removes any NaN values
    data2 = dataframe[data2name].dropna()




    # Perform Shapiro-Wilk test
    stat, p_value = stats.shapiro(data1)

    # Display results
    print(f"Shapiro-Wilk Test statistic: {stat}")
    print(f"P-value: {p_value}")

    # Check if the data is normally distributed (typically, p-value > 0.05)
    if p_value > 0.05:
        print("The data is normally distributed.")
    else:
        print("The data is not normally distributed.")





    # Perform two-sample t-test
    t_statistic, t_p_value = stats.ttest_ind(data1, data2)

    # Display T-Test results
    print(f"\nT-Test between ", data1name, " and ", data2name, ":")
    print(f"T-Test statistic: {t_statistic}")
    print(f"P-value: {t_p_value}")

    # Interpret the result
    if t_p_value > 0.05:
        print("There is no significant difference between the means of ", data1name, " and ", data2name)
    else:
        print("There is a significant difference between the means of ", data1name, " and ", data2name)







except Exception as e:
    print(f"An error ocurred: {e}")

