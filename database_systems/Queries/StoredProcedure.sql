CREATE PROCEDURE `create_organisation` (organisation varchar(55), country varchar(55))
BEGIN
insert into organisation(organisation) values (organisation);
SELECT 
    organisation_id
INTO @o_id FROM
    organisation;
SELECT 
    country_id
INTO @c_id FROM
    country;
insert into country_organisation(organisation_id, country_id) values(@o_id, @c_id);
END
