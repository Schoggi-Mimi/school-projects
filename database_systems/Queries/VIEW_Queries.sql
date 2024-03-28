CREATE VIEW water
AS SELECT cn.country, period, percentage_water, iso_a3 
FROM drinking_water_supply AS dw 
INNER JOIN country AS cn ON dw.country_ID = cn.country_id;

CREATE VIEW life
AS SELECT cn.country, period, percentage_life, iso_a3 
FROM life_expectancy AS le 
INNER JOIN country AS cn ON le.country_ID = cn.country_id;

CREATE VIEW infant
AS SELECT cn.country, period, percentage_infant, iso_a3 
FROM infant_mortality_rate AS im 
INNER JOIN country AS cn ON im.country_ID = cn.country_id;

CREATE VIEW water_life_infant 
AS SELECT cn.population_estimate, continent.continent, cn.iso_a3, cn.gdp_estimate, 
cn.country, dw.period, dw.percentage_water, le.percentage_life, im.percentage_infant 
FROM country AS cn 
LEFT JOIN continent ON cn.continent_ID = continent.continent_ID
LEFT JOIN drinking_water_supply AS dw ON cn.country_ID = dw.country_ID
LEFT JOIN infant_mortality_rate AS im ON dw.country_ID = im.country_ID 
AND dw.period = im.period
INNER JOIN life_expectancy AS le ON im.country_ID = le.country_ID 
AND im.period = le.period;

CREATE VIEW water_life 
AS SELECT cn.population_estimate, continent.continent, cn.iso_a3, 
cn.gdp_estimate, cn.country, dw.period, dw.percentage_water, le.percentage_life
FROM country AS cn
LEFT JOIN continent ON cn.continent_ID = continent.continent_ID
LEFT JOIN drinking_water_supply AS dw ON cn.country_ID = dw.country_ID
INNER JOIN life_expectancy AS le ON dw.country_ID = le.country_ID 
AND dw.period = le.period;

CREATE VIEW water_infant 
AS SELECT cn.population_estimate, continent.continent, cn.iso_a3, 
cn.gdp_estimate, cn.country, dw.period, dw.percentage_water, im.percentage_infant 
FROM country AS cn 
LEFT JOIN continent ON cn.continent_ID = continent.continent_ID
LEFT JOIN drinking_water_supply AS dw ON cn.country_ID = dw.country_ID
LEFT JOIN infant_mortality_rate AS im ON dw.country_ID = im.country_ID 
AND dw.period = im.period;

CREATE VIEW choose_switzerland
AS SELECT wln.population_estimate, wln.continent, wln.iso_a3, wln.gdp_estimate, 
wln.country, wln.period, wln.percentage_water, wln.percentage_life, wln.percentage_infant
FROM water_life_infant AS wln
WHERE wln.country = 'Switzerland';

CREATE VIEW organisation_country
AS SELECT cn.population_estimate, continent.continent, cn.iso_a3, cn.gdp_estimate, 
cn.country, og.organisation, wl.percentage_water, wl.percentage_life, wl.period
FROM country_organisation AS co 
LEFT JOIN country AS cn ON co.country_id = cn.country_id 
LEFT JOIN organisation AS og ON co.organisation_id = og.organisation_id
LEFT JOIN continent ON cn.continent_id = continent.continent_id
INNER JOIN water_life AS wl ON cn.iso_a3 = wl.iso_a3;