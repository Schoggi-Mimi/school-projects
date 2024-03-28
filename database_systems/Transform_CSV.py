import pandas as pd
import geopandas as gpd


def import_data(data: str) -> pd.DataFrame:
    return pd.read_csv(data)


def export_data(data: pd.DataFrame, name: str):
    data.to_csv('../csv/transform/' + name + '.csv', index=False, sep=',')


def clean(df):
    df['Name'] = df['Name'].str.replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df['Name'] = df['Name'].str.replace('Viet Nam', 'Vietnam')
    df['Name'] = df['Name'].str.replace(
        'Democratic Republic of the Congo', 'Dem. Rep. Congo')

    df['Name'] = df['Name'].str.replace('Republic of Korea', 'South Korea')
    df['Name'] = df['Name'].str.replace(
        "Democratic People's South Korea", 'North Korea')

    df['Name'] = df['Name'].str.replace("Lao People's Democratic Republic", 'Laos')
    df['Name'] = df['Name'].str.replace("Russian Federation", 'Russia')
    df['Name'] = df['Name'].str.replace("Equatorial Guinea", 'Eq. Guinea')
    df['Name'] = df['Name'].str.replace("Dominican Republic", 'Dominican Rep.')
    df['Name'] = df['Name'].str.replace("South Sudan", 'S. Sudan')
    df['Name'] = df['Name'].str.replace("Côte d’Ivoire", "Côte d'Ivoire")
    df['Name'] = df['Name'].str.replace(
        "United Republic of Tanzania", 'Tanzania')
    df['Name'] = df['Name'].str.replace(
        "Central African Republic", 'Central African Rep.')
    df['Name'] = df['Name'].str.replace("Syrian Arab Republic", 'Syria')
    df['Name'] = df['Name'].str.replace(
        "The former Yugoslav Republic of Macedonia", 'Macedonia')
    df['Name'] = df['Name'].str.replace(
        "Bosnia and Herzegovina", 'Bosnia and Herz.')
    df['Name'] = df['Name'].str.replace("Republic of Moldova", 'Moldova')
    df['Name'] = df['Name'].str.replace("Solomon Islands", 'Solomon Is.')
    df['Name'] = df['Name'].str.replace("Brunei Darussalam", 'Brunei')
    df['Name'] = df['Name'].str.replace("Eswatini", 'eSwatini')

    df.loc[df['Name'].str.startswith('Iran'), 'Name'] = 'Iran'
    df.loc[df['Name'].str.startswith('Venezuela'), 'Name'] = 'Venezuela'
    df.loc[df['Name'].str.startswith('Bolivia'), 'Name'] = 'Bolivia'
    df.loc[df['Name'].str.startswith('Micronesia'), 'Name'] = 'Micronesia'
    df.loc[df['Name'].str.startswith('Sudan'), 'Name'] = 'Sudan'

    # df.drop(['name'], axis = 1, inplace = True)


if __name__ == '__main__':
    basic_drinking_water = import_data('../csv/basicDrinkingWaterServices.csv')
    life_expectancy_at_birth = import_data('../csv/lifeExpectancyAtBirth.csv')
    infant_mortality_rate = import_data('../csv/infantMortalityRate.csv')
    world = gpd.read_file('../csv/ne_10m_admin_0_countries.shp')

    basic_drinking_water = basic_drinking_water.rename(columns={"Location": "Name"})
    life_expectancy_at_birth = life_expectancy_at_birth.rename(columns={"Location": "Name"})
    infant_mortality_rate = infant_mortality_rate.rename(columns={"Location": "Name"})

    tables = [basic_drinking_water, life_expectancy_at_birth, infant_mortality_rate]
    for table in tables:
        clean(table)

    life_expectancy_at_birth = life_expectancy_at_birth[life_expectancy_at_birth["Dim1"].str.contains("Male") == False]
    life_expectancy_at_birth = life_expectancy_at_birth[
        life_expectancy_at_birth["Dim1"].str.contains("Female") == False]

    infant_mortality_rate = infant_mortality_rate[infant_mortality_rate["Dim1"].str.contains("Male") == False]
    infant_mortality_rate = infant_mortality_rate[infant_mortality_rate["Dim1"].str.contains("Female") == False]

    tables = []

    world_map = world[["NAME", "POP_EST", "GDP_MD", "ISO_A3_EH", "CONTINENT"]]
    world_map = world_map.rename(
        columns={"NAME": "Name", "POP_EST": "pop_est", "GDP_MD": "gpd_md_est", "ISO_A3_EH": "iso_a3",
                 "CONTINENT": "continent"})

    continents = world_map.continent.unique()
    continents = pd.DataFrame(continents, columns=['Name']).sort_values('Name')
    print(continents)
    tables.append((continents, 'continents'))

    continent_codes = world_map.continent.astype('category')
    continent_map = {k: v for k, v in zip(world_map.continent, continent_codes.cat.codes + 1)}

    countries = basic_drinking_water.Name.unique()
    countries = pd.DataFrame(countries, columns=['Name'])
    clean(countries)
    countries = countries.merge(world_map, how='left', left_on='Name', right_on='Name')
    print(countries)
    countries.continent = countries.continent.map(continent_map)
    print(countries, continent_map)
    countries = countries.dropna().reset_index(drop=True)
    countries.continent = countries.continent.astype(int)
    tables.append((countries, 'countries'))

    country_codes = countries.Name.astype('category')
    country_map = {k: v for k, v in zip(countries.Name, country_codes.cat.codes + 1)}

    drinking_water_supply = basic_drinking_water[['Name', 'Period', 'First Tooltip']]
    drinking_water_supply.Name = drinking_water_supply.Name.map(country_map)
    drinking_water_supply = drinking_water_supply.dropna().reset_index(drop=True)
    tables.append((drinking_water_supply, 'drinking_water_supply'))

    infant_death_rate = infant_mortality_rate[['Name', 'Period', 'First Tooltip']]
    infant_death_rate['First Tooltip'] = infant_death_rate['First Tooltip'].str.extract(r'(^\d+\.\d+)')
    infant_death_rate['First Tooltip'] = infant_death_rate['First Tooltip'].astype(float)
    infant_death_rate.Name = infant_death_rate.Name.map(country_map)
    infant_death_rate = infant_death_rate.dropna().reset_index(drop=True)
    infant_death_rate['Name'] = infant_death_rate['Name'].astype(int)
    tables.append((infant_death_rate, 'infant_death_rate'))

    life_expectancy = life_expectancy_at_birth[['Name', 'Period', 'First Tooltip']]
    life_expectancy.Name = life_expectancy.Name.map(country_map)
    tables.append((life_expectancy, 'life_expectancy'))

    for table in tables:
        export_data(table[0], table[1])
