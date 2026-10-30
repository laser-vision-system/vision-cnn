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
/// The data type used for intensit measurement.
pub type IntensityType = f32;

/// Hide these values from submodules.
mod dimensions {
	#![allow(dead_code, reason = "Compile-time selection")]

	pub(super) const SOURCE_IMG_HEIGHT: u32 = 480;
	pub(super) const SOURCE_IMG_WIDTH: u32 = 640;

	pub(super) const SOURCE_IMG_HEIGHT_HALF: u32 = SOURCE_IMG_HEIGHT
		.checked_div_euclid(2)
		.expect("Image dimensions are even");
	pub(super) const SOURCE_IMG_WIDTH_HALF: u32 = SOURCE_IMG_WIDTH
		.checked_div_euclid(2)
		.expect("Image dimensions are even");

	pub(super) const SOURCE_IMG_HEIGHT_QUARTER: u32 = SOURCE_IMG_HEIGHT
		.checked_div_euclid(4)
		.expect("Image dimensions are even");
	pub(super) const SOURCE_IMG_WIDTH_QUARTER: u32 = SOURCE_IMG_WIDTH
		.checked_div_euclid(4)
		.expect("Image dimensions are even");

	pub(super) const SOURCE_IMG_HEIGHT_EIGHTH: u32 = SOURCE_IMG_HEIGHT
		.checked_div_euclid(8)
		.expect("Image dimensions are even");
	pub(super) const SOURCE_IMG_WIDTH_EIGHTH: u32 = SOURCE_IMG_WIDTH
		.checked_div_euclid(8)
		.expect("Image dimensions are even");

	pub(super) const SOURCE_IMG_HEIGHT_SIXTEENTH: u32 = SOURCE_IMG_HEIGHT
		.checked_div_euclid(8)
		.expect("Image dimensions are even");
	pub(super) const SOURCE_IMG_WIDTH_SIXTEENTH: u32 = SOURCE_IMG_WIDTH
		.checked_div_euclid(8)
		.expect("Image dimensions are even");

	pub(super) const DESIRED_IMG_SQUARE: u32 = 60;
}

const IMG_CHANNELS_LUMA: usize = 1;
const IMG_CHANNELS_RGB: usize = 3;

const IMG_HEIGHT: u32 = dimensions::DESIRED_IMG_SQUARE;
const IMG_WIDTH: u32 = dimensions::DESIRED_IMG_SQUARE;

const IMG_PIXELS: usize = (IMG_HEIGHT * IMG_WIDTH) as usize;
