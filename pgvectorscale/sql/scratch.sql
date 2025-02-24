-- Demo script for label filtering

-- Create a table to hold region names.

-- For simplicity the `name` field is just a string, but this table
-- can contain pretty much anything, so long as there is a numeric
-- primary key.  
CREATE TABLE regions (
    id serial PRIMARY KEY,
    name text UNIQUE NOT NULL
);

INSERT INTO regions (name) VALUES
    ('us-east'),
    ('us-west'),
    ('eu-central'),
    ('eu-west'),
    ('ap-southeast'),
    ('ap-south');

-- Create a table to hold the vectors and labels.
CREATE TABLE advertisements (
    id serial PRIMARY KEY,
    embedding vector(768),
    target_regions integer[],
    content text
);

-- Generate 1M vectors, each with 4 labels
INSERT INTO advertisements (embedding, target_regions, content)
SELECT
    *
FROM (
    SELECT
        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding,
        random_int_array(6, 4) AS target_regions,
        md5(random()::text) AS content
    FROM
        generate_series(1, 768 * 1000000) i
    GROUP BY
        i % 1000000) g;

CREATE INDEX ON advertisements USING diskann (embedding vector_l2_ops, target_regions);

-- CREATE
-- OR REPLACE FUNCTION bench (query text, iterations integer DEFAULT 100) RETURNS TABLE (
--     avg double precision,
--     min double precision,
--     q1 double precision,
--     median double precision,
--     q3 double precision,
--     p95 double precision,
--     max double precision
CREATE
OR REPLACE FUNCTION bench (query text, iterations integer DEFAULT 100) RETURNS TABLE (
    avg double precision,
    min double precision,
    q1 double precision,
    median double precision,
    q3 double precision,
    p95 double precision,
    max double precision
)  LANGUAGE plpgsql
AS $function$
DECLARE
   _start TIMESTAMPTZ;
   _end TIMESTAMPTZ;
   _delta DOUBLE PRECISION;
BEGIN
   CREATE TEMP TABLE IF NOT EXISTS _bench_results (
       elapsed DOUBLE PRECISION
   );

   -- Warm the cache
   FOR i IN 1..5 LOOP
     EXECUTE query;
   END LOOP;

   -- Run test and collect elapsed time into _bench_results table
   FOR i IN 1..iterations LOOP
     _start = clock_timestamp();
     EXECUTE query;
     _end = clock_timestamp();
     _delta = 1000 * ( extract(epoch from _end) - extract(epoch from _start) );
     INSERT INTO _bench_results VALUES (_delta);
   END LOOP;

   RETURN QUERY SELECT
     avg(elapsed),
     min(elapsed),
     percentile_cont(0.25) WITHIN GROUP (ORDER BY elapsed),
     percentile_cont(0.5) WITHIN GROUP (ORDER BY elapsed),
     percentile_cont(0.75) WITHIN GROUP (ORDER BY elapsed),
     percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed),
     max(elapsed)
     FROM _bench_results;
   DROP TABLE IF EXISTS _bench_results;

 END
$function$;

CREATE TABLE
    items (embedding vector(8), labels integer[]);

-- Helper function to generate random integer arrays
-- from the range [1, $1] of length $2
CREATE OR REPLACE FUNCTION random_int_array(int, int)
RETURNS integer[] language sql as
$$
    SELECT array_agg(DISTINCT (round(random()* ($1-1))+1)::int)
    FROM generate_series(1, $2)
$$;

select
    setseed (0.5);

-- generate 300 vectors, each with 4 labels
INSERT INTO items (embedding, labels)
SELECT
    *
FROM (
    SELECT
        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding,
        random_int_array(10, 4) AS labels
    FROM
        generate_series(1, 8 * 300) i
    GROUP BY
        i % 300) g;

CREATE INDEX ON items USING diskann (embedding vector_cosine_ops, labels);
SET enable_seqscan = 0;


-- perform index scans on the vectors
SELECT
    *
FROM
    items
WHERE
    labels && '{9, 10}'
ORDER BY
    embedding <=> (
        SELECT
            ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
FROM generate_series(1, 8))
LIMIT 10;


-- SETUP
-- gist_960_euclidean (960 dimensions, euclidean/L2 distance -- supported since v0.5.0,
--                     1M vectors, 960*4 bytes per vector*1M vectors ~ 3.6GB raw data)
-- plus random labels for each vector in the range (1,255), 8 per vector
-- TJ's laptop with timescale-tuned config

-- No filtering, with diskann: around 25ms
SELECT
    id, labels
FROM
    gist_960_euclidean
ORDER BY
    embedding <-> (
        SELECT
            ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
FROM generate_series(1, 960))
LIMIT 100;

-- Post-filtering, with diskann: around 250ms
SELECT
    id, labels
FROM
    gist_960_euclidean
WHERE
    labels @> '{1, 9, 10}'
ORDER BY
    embedding <-> (
        SELECT
            ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
FROM generate_series(1, 960))
LIMIT 10;

-- Filtered diskann:
SELECT
    id, labels
FROM
    gist_960_euclidean
WHERE
    labels && (SELECT array_agg(id) FROM regions WHERE name = 'us-east' OR name = 'us-west')
ORDER BY
    embedding <-> (
        SELECT
            ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
FROM generate_series(1, 960))
LIMIT 10;


-- Escaped, no filter
SELECT
    id, labels
FROM
    gist_960_euclidean
ORDER BY
    embedding <-> (
        SELECT
            (''['' || array_to_string(array_agg(random()), '','', ''0'') || '']'')::vector AS embedding
FROM generate_series(1, 960))
LIMIT 100;


-- Escaped, with filter
select * from bench('
SELECT
    id, labels
FROM
    gist_960_euclidean
WHERE
    labels && ''{1, 9, 10}''
ORDER BY
    embedding <-> (
        SELECT
            (''['' || array_to_string(array_agg(random()), '','', ''0'') || '']'')::vector AS embedding
FROM generate_series(1, 960))
LIMIT 100;
');

-- Escaped, with post-filter
SELECT
    id, labels
FROM
    gist_960_euclidean
WHERE
    labels @> ''{1, 9, 10}''
ORDER BY
    embedding <-> (
        SELECT
            (''['' || array_to_string(array_agg(random()), '','', ''0'') || '']'')::vector AS embedding
FROM generate_series(1, 960))
LIMIT 100;


SELECT
    id, labels
FROM
    hnsw_gist_960_euclidean
ORDER BY
    binary_quantize(embedding)::bit(960) <~>
        SELECT
            ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector)::bit(960) AS embedding
FROM generate_series(1, 960))
LIMIT 100;
