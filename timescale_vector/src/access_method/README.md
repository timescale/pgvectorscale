# On Disk Layout

Meta Page
- basic metadata
- future proof
- page 0
- start_node tid

Graph pages
-- start node first
- foreach node
-- vector for node
-- array of tids of neighbors
-- array of distances?

- in "special area"
-- bitmap of deletes?

