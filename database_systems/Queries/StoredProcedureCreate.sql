CREATE DEFINER=`admin`@`%` PROCEDURE `create_organisation`(in porganisation varchar(55),in pcountry varchar(55))
BEGIN
declare c_id int;
declare o_id int;

set autocommit = 0;

insert into organisation(organisation) values (porganisation);
SELECT 
    @o_id:=organisation_id, @c_id:=country_id
FROM
    organisation,
    country
WHERE
    organisation = porganisation
        AND country = pcountry;
insert into country_organisation(organisation_id, country_id) values (@o_id, @c_id);
END