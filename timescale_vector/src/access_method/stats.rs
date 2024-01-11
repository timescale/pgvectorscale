use std::time::Instant;

pub trait StatsNodeRead {
    fn record_read(&mut self);
}

pub trait StatsNodeModify {
    fn record_modify(&mut self);
}

pub trait StatsNodeWrite {
    fn record_write(&mut self);
}

pub trait StatsDistanceComparison {
    fn record_full_distance_comparison(&mut self);
    fn record_quantized_distance_comparison(&mut self);
}

#[derive(Debug)]
pub struct PruneNeighborStats {
    pub calls: usize,
    pub distance_comparisons: usize,
    pub node_reads: usize,
    pub node_modify: usize,
    pub num_neighbors_before_prune: usize,
    pub num_neighbors_after_prune: usize,
}

impl PruneNeighborStats {
    pub fn new() -> Self {
        PruneNeighborStats {
            calls: 0,
            distance_comparisons: 0,
            node_reads: 0,
            node_modify: 0,
            num_neighbors_before_prune: 0,
            num_neighbors_after_prune: 0,
        }
    }
}

impl StatsDistanceComparison for PruneNeighborStats {
    fn record_full_distance_comparison(&mut self) {
        self.distance_comparisons += 1;
    }

    fn record_quantized_distance_comparison(&mut self) {
        pgrx::error!("Should not use quantized distance comparisons during pruning");
    }
}

impl StatsNodeRead for PruneNeighborStats {
    fn record_read(&mut self) {
        self.node_reads += 1;
    }
}

impl StatsNodeModify for PruneNeighborStats {
    fn record_modify(&mut self) {
        self.node_modify += 1;
    }
}

#[derive(Debug)]
pub struct GreedySearchStats {
    calls: usize,
    full_distance_comparisons: usize,
    node_reads: usize,
    quantized_distance_comparisons: usize,
    quantizer_stats: Option<QuantizerStats>,
}

impl GreedySearchStats {
    pub fn new() -> Self {
        GreedySearchStats {
            calls: 0,
            full_distance_comparisons: 0,
            node_reads: 0,
            quantized_distance_comparisons: 0,
            quantizer_stats: None,
        }
    }

    pub fn set_quantizer_stats(&mut self, quantizer_stats: QuantizerStats) {
        self.quantizer_stats = Some(quantizer_stats);
    }

    pub fn get_calls(&self) -> usize {
        self.calls
    }

    pub fn get_node_reads(&self) -> usize {
        self.node_reads
    }

    pub fn get_total_distance_comparisons(&self) -> usize {
        self.full_distance_comparisons + self.quantized_distance_comparisons
    }

    pub fn get_quantized_distance_comparisons(&self) -> usize {
        self.quantized_distance_comparisons
    }

    pub fn get_full_distance_comparisons(&self) -> usize {
        self.full_distance_comparisons
    }

    pub fn record_call(&mut self) {
        self.calls += 1;
    }
}

impl StatsNodeRead for GreedySearchStats {
    fn record_read(&mut self) {
        self.node_reads += 1;
    }
}

impl StatsDistanceComparison for GreedySearchStats {
    fn record_full_distance_comparison(&mut self) {
        self.full_distance_comparisons += 1;
    }

    fn record_quantized_distance_comparison(&mut self) {
        self.quantized_distance_comparisons += 1;
    }
}

#[derive(Debug)]
pub struct QuantizerStats {
    pub node_reads: usize,
    pub node_writes: usize,
}

impl QuantizerStats {
    pub fn new() -> Self {
        QuantizerStats {
            node_reads: 0,
            node_writes: 0,
        }
    }
}

impl StatsNodeRead for QuantizerStats {
    fn record_read(&mut self) {
        self.node_reads += 1;
    }
}

impl StatsNodeWrite for QuantizerStats {
    fn record_write(&mut self) {
        self.node_writes += 1;
    }
}
#[derive(Debug)]
pub struct InsertStats {
    pub prune_neighbor_stats: PruneNeighborStats,
    pub greedy_search_stats: GreedySearchStats,
    pub quantizer_stats: QuantizerStats,
    pub node_reads: usize,
    pub node_modify: usize,
    pub node_writes: usize,
}

impl InsertStats {
    pub fn new() -> Self {
        return InsertStats {
            prune_neighbor_stats: PruneNeighborStats::new(),
            greedy_search_stats: GreedySearchStats::new(),
            quantizer_stats: QuantizerStats::new(),
            node_reads: 0,
            node_modify: 0,
            node_writes: 0,
        };
    }
}

impl StatsNodeRead for InsertStats {
    fn record_read(&mut self) {
        self.node_reads += 1;
    }
}

impl StatsNodeModify for InsertStats {
    fn record_modify(&mut self) {
        self.node_modify += 1;
    }
}

impl StatsNodeWrite for InsertStats {
    fn record_write(&mut self) {
        self.node_writes += 1;
    }
}

pub struct WriteStats {
    pub started: Instant,
    pub num_nodes: usize,
    pub nodes_read: usize,
    pub nodes_modified: usize,
    pub nodes_written: usize,
    pub prune_stats: PruneNeighborStats,
    pub num_neighbors: usize,
}

impl WriteStats {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            num_nodes: 0,
            prune_stats: PruneNeighborStats::new(),
            num_neighbors: 0,
            nodes_read: 0,
            nodes_modified: 0,
            nodes_written: 0,
        }
    }
}

impl StatsNodeRead for WriteStats {
    fn record_read(&mut self) {
        self.nodes_read += 1;
    }
}

impl StatsNodeModify for WriteStats {
    fn record_modify(&mut self) {
        self.nodes_modified += 1;
    }
}

impl StatsNodeWrite for WriteStats {
    fn record_write(&mut self) {
        self.nodes_written += 1;
    }
}
