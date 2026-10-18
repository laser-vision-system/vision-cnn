//! Source loading, dataset mapping, and batching.

pub mod batching;
pub mod mapping;
pub mod sources;

pub use self::{
	batching::{
		AugmentedLaserBatcher,
		LaserBatch,
		LaserBatcher,
	},
	mapping::{
		LaserItemMapper,
		LoadedDataset,
		LoadedLaserItem,
	},
	sources::{
		DatasetSource,
		DatasetSplits,
	},
};

/// The image buffer representation that is supported by the current model.
pub type ImageType = image::GrayImage;
