-- Filtering based on regions
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

-- Target regions are arrays of foreign keys in the regions table
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

-- Create filtered diskann index.  This will support
-- (a) Euclidean/L2 distance queries on `embedding` via `<->`
-- (b) Filtering on `target_regions` via `&&` overlap operator
CREATE INDEX ON advertisements USING diskann (embedding vector_l2_ops, target_regions);

-- Query non-indexed table
SELECT
    id, target_regions
FROM
    advertisements
WHERE
    target_regions && (SELECT array_agg(id) FROM regions WHERE name = 'us-east' OR name = 'us-west')
ORDER BY
    embedding <-> (
        SELECT
            ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
FROM generate_series(1, 768))
LIMIT 10;


-- Query for random vector with filter condition
SELECT
    id, target_regions
FROM
    advertisements_diskann
WHERE
    target_regions && (SELECT array_agg(id) FROM regions WHERE name = 'us-east' OR name = 'us-west')
ORDER BY
    embedding <-> (
        SELECT
            ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
FROM generate_series(1, 768))
LIMIT 10;


-- Query for random vector, force post-filtering by using non-supported operator `@>`
SELECT
    id, target_regions
FROM
    advertisements_diskann
WHERE
    target_regions @> '{1}'
ORDER BY
    embedding <-> (
        SELECT
            ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
FROM generate_series(1, 768))
LIMIT 10;

