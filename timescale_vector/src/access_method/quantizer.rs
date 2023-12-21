use super::{bq::BqQuantizer, pq::PqQuantizer};

/*pub trait Quantizer {
    fn initialize_node(&self, node: &mut Node, meta_page: &MetaPage);
    fn start_training(&mut self, meta_page: &super::meta_page::MetaPage);
    fn add_sample(&mut self, sample: Vec<f32>);
    fn finish_training(&mut self);
}*/

pub enum Quantizer {
    BQ(BqQuantizer),
    PQ(PqQuantizer),
    None,
}

impl Quantizer {
    pub fn is_some(&self) -> bool {
        match self {
            Quantizer::None => false,
            _ => true,
        }
    }
}
