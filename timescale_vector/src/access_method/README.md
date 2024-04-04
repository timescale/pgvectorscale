# Graph

The graph abstraction implements the 2 primary algorithms:
- greedy_search - which finds the index nodes closest to a given query
- prune_neighbors - which reduces the number of neighbors assigned to a particular nodes in order to fit within the num_neighbor limit.

Graph also implements the insertion algoritm for when a node needs to be added to the graph. Insertion is mostly a combination of the 2 algorithms above, greedy_search and prune_neighbors.

We support multiple storage layouts (described below). The logic in graph is works on an abstract storage object and is not concerned with the particular storage implementation.
Thus, any logic that needs to differ between storage implementations has to fall within the responsibility of the storage object and not be included in graph.

## Greedy search

Refer to the DiskANN paper for an overview. The greedy search algorithm works by traversing the graph to find the closest nodes to a given query. It does this by:
- starting with a set (right now implemented as just one) initial nodes (called init_ids).
- iteratively:
- -



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

