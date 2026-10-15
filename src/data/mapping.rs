#![expect(
	clippy::expect_used,
	reason = "Impossible to handle error inside mapper"
)]

use burn::data::dataset::transform::Mapper;
use dataset_loader::item::{
	ImageType,
	LaserItem,
	NumericLike,
};

/// The data-structure used post-mapping.
#[derive(Debug, Clone)]
pub struct LoadedLaserItem<T: NumericLike> {
	pub image: ImageType,
	pub image_path: String,
	pub intensity: T,
	pub position: Option<String>,
	pub source_fields: Vec<(&'static str, String)>,
}

/// The unit struct with the [`Mapper`] trait.
pub struct LaserItemMapper;

impl<I, T> Mapper<I, LoadedLaserItem<T>> for LaserItemMapper
where
	I: LaserItem<IntensityType = T> + Send + Sync + Clone,
	T: NumericLike,
{
	fn map(
		&self,
		item: &I,
	) -> LoadedLaserItem<T> {
		LoadedLaserItem {
			image: item.load().expect("the image should not fail to load"),
			image_path: item.get_image_path().to_string(),
			intensity: *item.get_intensity(),
			position: item.get_position().map(str::to_owned),
			source_fields: item.source_fields(),
		}
	}
}
