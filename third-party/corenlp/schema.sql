-- Additional SQL things.

-- Search Index Triggers (https://www.postgresql.org/docs/current/static/textsearch-features.html#TEXTSEARCH-UPDATE-TRIGGERS)

CREATE TRIGGER sentence_searchable_trigger BEFORE INSERT OR UPDATE
    ON corenlp_sentence FOR EACH ROW EXECUTE PROCEDURE
    tsvector_update_trigger(searchable, 'pg_catalog.english', gloss);


-- Create functions because django textsearch is broken
CREATE FUNCTION ts_query_or(tsquery, tsquery) RETURNS tsquery AS 'SELECT $1 || $2;' LANGUAGE SQL IMMUTABLE RETURNS NULL ON NULL INPUT;
CREATE FUNCTION to_tsvector(tsvector) RETURNS tsvector AS 'SELECT $1;' LANGUAGE SQL IMMUTABLE RETURNS NULL ON NULL INPUT;
CREATE OPERATOR | (PROCEDURE=ts_query_or, LEFT_ARG=tsquery, RIGHT_ARG=tsquery, COMMUTATOR= |); 
