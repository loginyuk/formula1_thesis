import pandas as pd


def prepare_race(session):
    """
    Get laps, drivers and stints dataframes for the race
    """
    laps = session.laps
    laps['Location'] = session.event['Location']
    laps['Year'] = session.event.year
    laps['RoundNumber'] = session.event['RoundNumber']

    drivers = session.drivers
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})

    return laps, drivers, stints


def merge_weather(session, laps):
    """
    Merge weather data with laps data
    """
    df = laps.copy()

    weather = session.weather_data.copy()

    df = df.sort_values('Time')
    weather = weather.sort_values('Time')

    df_weather = pd.merge_asof(df, weather, on='Time', direction='backward')

    df_weather['LapTime_Sec'] = df_weather['LapTime'].dt.total_seconds()
    return df_weather


def get_pirelli_press_data(file_path):
    """
    Loads Pirelli press dataset from CSV
    """
    df = pd.read_csv(file_path)
    return df
