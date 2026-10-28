//! Mapping operations to load images from their low-level representations.

#![expect(
	clippy::expect_used,
	reason = "Impossible to handle error inside mapper"
)]

use burn::{
	data::dataset::{
		Dataset,
		transform::Mapper,
	},
	tensor::Element,
};
use dataset_loader::item::LaserItem;
use image::imageops::FilterType;

use super::{
	IMG_HEIGHT,
	IMG_WIDTH,
	ImageType,
	IntensityType,
};

/// The data-structure used post-mapping.
#[derive(Debug, Clone)]
pub struct LoadedLaserItem {
	/// The image data.
	pub image: ImageType,
	/// The path to the image.
	pub image_path: String,
	/// The intensity (label) of the item.
	pub intensity: IntensityType,
	/// Optional information about the position that the item was capture in.
	pub position: Option<String>,
	/// Source-specific details about this item, used for evaluation.
	pub source_fields: Vec<(String, String)>,
}

/// A dataset of mapped samples whose target scalar matches a Burn backend.
pub type LoadedDataset = Box<dyn Dataset<LoadedLaserItem>>;

/// Maps source items and converts their intensity to the backend float type.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaserItemMapper;

impl<I, T> Mapper<I, LoadedLaserItem> for LaserItemMapper
where
	T: Element,
	I: LaserItem<IntensityType = T> + Send + Sync + Clone,
	IntensityType: From<T>,
{
	fn map(
		&self,
		item: &I,
	) -> LoadedLaserItem {
		let image = item
			.load()
			.expect("the image should not fail to load")
			.resize_to_fill(IMG_WIDTH, IMG_HEIGHT, FilterType::Nearest)
			.into();
		LoadedLaserItem {
			image,
			image_path: item.get_image_path().to_string(),
			intensity: map_intensity(*item.get_intensity()),
			position: item.get_position().map(str::to_owned),
			source_fields: item.source_fields(),
		}
	}
}

fn map_intensity<T>(intensity: T) -> IntensityType
where
	T: Element,
	IntensityType: From<T>,
{
	intensity.into()
}

#[cfg(test)]
mod tests {

	use super::*;

	#[test]
	fn maps_compact_integer_intensity_to_backend_float_element() {
		let mapped = map_intensity(53_u8);

		assert_eq!(mapped, 53_f32);
	}
}
