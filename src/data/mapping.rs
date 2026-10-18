#![expect(
	clippy::expect_used,
	reason = "Impossible to handle error inside mapper"
)]

use std::marker::PhantomData;

use burn::{
	data::dataset::{
		Dataset,
		transform::Mapper,
	},
	prelude::Backend,
	tensor::{
		Element,
		ElementConversion as _,
	},
};
use dataset_loader::item::LaserItem;

use super::ImageType;

/// The data-structure used post-mapping.
#[derive(Debug, Clone)]
pub struct LoadedLaserItem<B: Backend> {
	pub image: ImageType,
	pub image_path: String,
	pub intensity: B::FloatElem,
	pub position: Option<String>,
	pub source_fields: Vec<(&'static str, String)>,
}

/// A dataset of mapped samples whose target scalar matches a Burn backend.
pub type LoadedDataset<B> = Box<dyn Dataset<LoadedLaserItem<B>>>;

/// Maps source items and converts their intensity to the backend float type.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaserItemMapper<B>(PhantomData<B>);

impl<B, I, T> Mapper<I, LoadedLaserItem<B>> for LaserItemMapper<B>
where
	B: Backend,
	I: LaserItem<IntensityType = T> + Send + Sync + Clone,
	T: Element,
{
	fn map(
		&self,
		item: &I,
	) -> LoadedLaserItem<B> {
		LoadedLaserItem {
			image: item
				.load()
				.expect("the image should not fail to load")
				.into(),
			image_path: item.get_image_path().to_string(),
			intensity: map_intensity::<B, _>(*item.get_intensity()),
			position: item.get_position().map(str::to_owned),
			source_fields: item.source_fields(),
		}
	}
}

fn map_intensity<B, T>(intensity: T) -> B::FloatElem
where
	B: Backend,
	T: Element,
{
	B::FloatElem::from_elem(intensity)
}

#[cfg(test)]
mod tests {
	use burn_flex::Flex;

	use super::*;

	#[test]
	fn maps_compact_integer_intensity_to_backend_float_element() {
		let mapped = map_intensity::<Flex, _>(53_u8);

		assert_eq!(mapped, 53_f32);
	}
}
