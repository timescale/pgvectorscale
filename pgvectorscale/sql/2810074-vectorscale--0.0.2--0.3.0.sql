/* 
This file is auto generated by pgrx.

The ordering of items is not stable, it is driven by a dependency graph.
*/

-- src/access_method/mod.rs:44
-- vectorscale::access_method::amhandler

    CREATE OR REPLACE FUNCTION diskann_amhandler(internal) RETURNS index_am_handler PARALLEL SAFE IMMUTABLE STRICT COST 0.0001 LANGUAGE c AS '$libdir/vectorscale-0.3.0', 'amhandler_wrapper';

    DO $$
    DECLARE
        c int;
    BEGIN
        SELECT count(*)
        INTO c
        FROM pg_catalog.pg_am a
        WHERE a.amname = 'diskann';

        IF c = 0 THEN
            CREATE ACCESS METHOD diskann TYPE INDEX HANDLER diskann_amhandler;
        END IF;
    END;
    $$;




-- src/access_method/mod.rs:89

DO $$
DECLARE
  c int;
BEGIN
    SELECT count(*)
    INTO c
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_cosine_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am  WHERE am.amname = 'diskann');

    IF c = 0 THEN
        CREATE OPERATOR CLASS vector_cosine_ops DEFAULT
        FOR TYPE vector USING diskann AS
	        OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops;
    END IF;
END;
$$