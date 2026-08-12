CREATE OR REPLACE FUNCTION diskann_amhandler(internal)
RETURNS index_am_handler
PARALLEL SAFE IMMUTABLE STRICT COST 0.0001
LANGUAGE c
AS 'vectorscale-0.9.1', 'amhandler_wrapper';

CREATE OR REPLACE FUNCTION distance_type_cosine()
RETURNS smallint
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c
AS 'vectorscale-0.9.1', 'distance_type_cosine_wrapper';

CREATE OR REPLACE FUNCTION distance_type_inner_product()
RETURNS smallint
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c
AS 'vectorscale-0.9.1', 'distance_type_inner_product_wrapper';

CREATE OR REPLACE FUNCTION distance_type_l2()
RETURNS smallint
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c
AS 'vectorscale-0.9.1', 'distance_type_l2_wrapper';

CREATE OR REPLACE FUNCTION smallint_array_overlap(
    "left" smallint[],
    "right" smallint[]
)
RETURNS bool
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c
AS 'vectorscale-0.9.1', 'smallint_array_overlap_wrapper';

DO $$
DECLARE
    expected_vector_type oid;
BEGIN
    SELECT t.oid
    INTO STRICT expected_vector_type
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_type t
      ON t.typnamespace = e.extnamespace
     AND t.typname = 'vector'
    WHERE e.extname = 'vector';

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_opclass c
        JOIN pg_catalog.pg_am am ON am.oid = c.opcmethod
        WHERE am.amname = 'diskann'
          AND c.opcnamespace = (
              SELECT oid
              FROM pg_catalog.pg_namespace
              WHERE nspname = '@extschema@'
          )
          AND c.opcname IN (
              'vector_cosine_ops',
              'vector_l2_ops',
              'vector_ip_ops'
          )
          AND c.opcintype IS DISTINCT FROM expected_vector_type
    ) THEN
        RAISE EXCEPTION
            'diskann: a vector operator class is not bound to pgvector''s vector type; drop the affected operator class and recreate the extension objects';
    END IF;
END;
$$;
