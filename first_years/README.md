copy over some other tables and delete later years so it's easy to run MAF


DELETE FROM observations where night > 365;
DELETE FROM observations where night > 730;
