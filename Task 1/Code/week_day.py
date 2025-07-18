import pandas as pd

try:
    # Cargar el archivo CSV
    dataframe = pd.read_csv('spotify-2023.csv', encoding='latin')  # Cambia el nombre del archivo según sea necesario

    # Mostrar las primeras filas del dataframe (opcional)
    print(dataframe.head())


    dataframe['release_date'] = pd.to_datetime(
        dataframe['released_year'].astype(str) + '-' + 
        dataframe['released_month'].astype(str) + '-' + 
        dataframe['released_day'].astype(str),
        errors='coerce'  # Convierte valores no válidos a NaT
    )

    if dataframe['release_date'].isnull().any():
        print("Not valid date numbers.")

    # Obtaining week day
    dataframe['day_of_week'] = dataframe['release_date'].dt.day_name()

    # Counting songs by week day
    songs_per_day = dataframe['day_of_week'].value_counts().reset_index()
    songs_per_day.columns = ['day_of_week', 'count']  # Renombrar columnas

    # Ordering
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    songs_per_day['day_of_week'] = pd.Categorical(songs_per_day['day_of_week'], categories=days_order, ordered=True)
    songs_per_day = songs_per_day.sort_values('day_of_week')


    print("\nNumber of songs published each day of the week:")
    print(songs_per_day)


except Exception as e:
    print(f"An error ocurred: {e}")
