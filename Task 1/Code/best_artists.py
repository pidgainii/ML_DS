import pandas as pd

try:

    dataframe = pd.read_csv('spotify-2023.csv', encoding='latin')

    dataframe['artist(s)_name'] = dataframe['artist(s)_name'].str.split(', ')
    exploded_df = dataframe.explode('artist(s)_name')

    artist_counts = exploded_df['artist(s)_name'].value_counts().reset_index()
    artist_counts.columns = ['artist(s)_name', 'count']  # Renombrar columnas

    top_artists = artist_counts.sort_values(by='count', ascending=False).head(10)

    print("\nThe artists that appear the most are:")
    print(top_artists)

except Exception as e:
    print(f"An error ocurred: {e}")
