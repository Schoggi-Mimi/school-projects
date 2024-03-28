CREATE USER developer IDENTIFIED BY 'q~vUYe@-gJ>5e2;{o}x!';
GRANT ALL ON dankdaten.* TO developer;
FLUSH PRIVILEGES;


CREATE USER dashboard IDENTIFIED BY 'p~bUYk@-fJ>4e7;{s}x@';

GRANT SELECT ON dankdaten.* TO dashboard;

GRANT SELECT 
	ON dankdaten.water_life_infant
	TO dashboard;
GRANT SELECT 
	ON dankdaten.water_life
	TO dashboard;
GRANT SELECT 
	ON dankdaten.water_infant
	TO dashboard;
GRANT SELECT 
	ON dankdaten.choose_switzerland
	TO dashboard;
GRANT SELECT 
	ON dankdaten.organisation_country
	TO dashboard;
GRANT SELECT 
	ON dankdaten.life
	TO dashboard;
GRANT SELECT 
	ON dankdaten.water
	TO dashboard;
GRANT SELECT 
	ON dankdaten.infant
	TO dashboard;

FLUSH PRIVILEGES;