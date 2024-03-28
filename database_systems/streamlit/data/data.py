import database.database_connection as db_connection
import pandas as pd

db = db_connection.init_connection()


def rename_database(df):
    return df.rename(columns={'population_estimate': 'Bevölkerung', 'continent': 'Kontinent', 'gdp_estimate': 'GDP',
                              'organisation': 'NPO', 'country': 'Land', 'period': 'Jahr',
                              'percentage_water': 'Trinkwasser Zugang', 'percentage_life': 'Lebenserwartung',
                              'percentage_infant': 'Säuglingssterberate', 'iso_a3': 'ISO'}, inplace=True)


def get_water_life_infant():
    df = pd.read_sql("SELECT * FROM dankdaten.water_life_infant", db)
    rename_database(df)
    return df


def get_water_life():
    df = pd.read_sql("SELECT * FROM dankdaten.water_life", db)
    rename_database(df)
    return df


def get_water_infant():
    df = pd.read_sql("SELECT * FROM dankdaten.water_infant", db)
    rename_database(df)
    return df


def get_switzerland():
    df = pd.read_sql("SELECT * FROM dankdaten.choose_switzerland", db)
    rename_database(df)
    return df


def get_organization():
    df = pd.read_sql('SELECT * FROM dankdaten.organisation_country', db)
    rename_database(df)
    return df


def get_drinking_water():
    df = pd.read_sql('SELECT * FROM dankdaten.water', db)
    rename_database(df)
    return df


def get_life_expectancy():
    df = pd.read_sql('SELECT * FROM dankdaten.life', db)
    rename_database(df)
    return df


def get_infant_death_rate():
    df = pd.read_sql('SELECT * FROM dankdaten.infant', db)
    rename_database(df)
    return df
