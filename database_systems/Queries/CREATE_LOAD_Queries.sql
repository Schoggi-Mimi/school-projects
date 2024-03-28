------------------------------------------------------------
-- create continent table
------------------------------------------------------------
CREATE TABLE continent
(
continent_ID INT NOT NULL AUTO_INCREMENT,
continent varchar(55),
PRIMARY KEY (continent_ID)
);
------------------------------------------------------------
-- load continent csv in table
------------------------------------------------------------
LOAD DATA LOCAL INFILE '/Users/oliver/HSLU/DBS/csv/transform/continents.csv'
INTO TABLE continent FIELDS ENCLOSED BY '"' ESCAPED BY '\\' LINES TERMINATED 
BY '\n' IGNORE 1 LINES (continent);
------------------------------------------------------------
-- create country table
------------------------------------------------------------
CREATE TABLE country
(
country_ID INT NOT NULL AUTO_INCREMENT,
country VARCHAR(35),
population_estimate INT,
gdp_estimate INT,
iso_a3 VARCHAR(3),
continent_ID INT,
PRIMARY KEY (country_ID),
FOREIGN KEY (continent_ID) REFERENCES continent(continent_ID)
);
------------------------------------------------------------
-- load country csv in table
------------------------------------------------------------
LOAD DATA LOCAL INFILE '/Users/oliver/HSLU/DBS/csv/transform/countries.csv'
INTO TABLE country FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' ESCAPED 
BY '\\' LINES TERMINATED BY '\n' IGNORE 1 LINES (country, population_estimate, 
gdp_estimate, iso_a3, continent_ID);
------------------------------------------------------------
-- create life_expectancy table
------------------------------------------------------------
CREATE TABLE life_expectancy
(
country_ID INT,
period YEAR,
percentage_life DECIMAL(5,2),
PRIMARY KEY (country_ID, period),
FOREIGN KEY (country_ID) REFERENCES country(country_ID));
------------------------------------------------------------
-- load life expectancy csv in table
------------------------------------------------------------
LOAD DATA LOCAL INFILE '/Users/oliver/HSLU/DBS/csv/transform/life_expectancy.csv'
INTO TABLE life_expectancy FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED 
BY '"' ESCAPED BY '\\' LINES TERMINATED BY '\n' 
IGNORE 1 LINES (country_ID, period, percentage_life);
------------------------------------------------------------
-- create drinking_water_supply table
------------------------------------------------------------
CREATE TABLE drinking_water_supply
(
country_ID INT,
period YEAR,
percentage_water DECIMAL(5,2),
PRIMARY KEY (country_ID, period),
FOREIGN KEY (country_ID) REFERENCES country(country_ID));
------------------------------------------------------------
-- load drinking water supply csv in table
------------------------------------------------------------
LOAD DATA LOCAL INFILE '/Users/oliver/HSLU/DBS/csv/transform/drinking_water_supply.csv'
INTO TABLE drinking_water_supply FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED 
BY '"' ESCAPED BY '\\' LINES TERMINATED BY '\n' 
IGNORE 1 LINES (country_ID, period, percentage_water);
------------------------------------------------------------
-- create infant_mortality_rate table
------------------------------------------------------------
CREATE TABLE infant_mortality_rate
(
country_ID INT,
period YEAR,
percentage_infant DECIMAL(5,2),
PRIMARY KEY (country_ID, period),
FOREIGN KEY (country_ID) REFERENCES country(country_ID));
------------------------------------------------------------
-- load infant mortality rate csv in table
------------------------------------------------------------
LOAD DATA LOCAL INFILE '/Users/oliver/HSLU/DBS/csv/transform/infant_death_rate.csv'
INTO TABLE infant_mortality_rate FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED 
BY '"' ESCAPED BY '\\' LINES TERMINATED BY '\n' 
IGNORE 1 LINES (country_ID, period, percentage_infant);
------------------------------------------------------------
-- create organisation table
------------------------------------------------------------
CREATE TABLE organisation(
organisation_id INT NOT NULL AUTO_INCREMENT,
organisation VARCHAR(55) NOT NULL,
PRIMARY KEY (organisation_id)
);
------------------------------------------------------------
-- insert top 5 organisations in table
------------------------------------------------------------
INSERT INTO organisation(organisation)
VALUES ('Blood:Water'), ('Pure Water for the World'), 
('water for good'), ('splash'), ('Lifewater');
------------------------------------------------------------
-- create country_organisation table
------------------------------------------------------------
CREATE TABLE country_organisation(
organisation_id INT NOT NULL,
country_id INT NOT NULL,
PRIMARY KEY (organisation_id, country_id),
FOREIGN KEY (organisation_id) REFERENCES organisation(organisation_id),
FOREIGN KEY (country_id) REFERENCES country(country_id)
);
------------------------------------------------------------
-- insert country id from organisation in table
------------------------------------------------------------
INSERT INTO country_organisation(organisation_id, country_id)
VALUES (1, 60), (1, 90), (1, 103), (1, 181);

INSERT INTO country_organisation(organisation_id, country_id)
VALUES (2, 75), (2, 76);

INSERT INTO country_organisation(organisation_id, country_id)
VALUES (3, 33);

INSERT INTO country_organisation(organisation_id, country_id)
VALUES (4, 14), (4, 30), (4, 36), (4, 60), (4, 79), (4, 121), 
(4, 172), (4, 191);

INSERT INTO country_organisation(organisation_id, country_id)
VALUES (5, 30), (5, 60), (5, 181), (5, 185);

