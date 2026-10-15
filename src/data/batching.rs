use std::{
	convert::identity,
	sync::{
		Arc,
		Mutex,
		PoisonError,
	},
};

use burn::{
	data::dataloader::batcher::Batcher,
	prelude::*,
};
use dataset_loader::item::{
	ImageType,
	NumericLike,
};
use rand::{
	SeedableRng as _,
	rngs::StdRng,
};

use super::{
	ImageStep,
	IntensityValType,
};
use crate::data::LoadedLaserItem;

// const IMG_CHANNELS: u32 = 3;
const IMG_HEIGHT: u32 = 480;
const IMG_WIDTH: u32 = 640;
const IMG_PIXELS: usize = (IMG_HEIGHT * IMG_WIDTH) as usize;

#[derive(Clone, Debug)]
pub struct LaserBatch<B>
where
	B: Backend,
{
	pub images: Tensor<B, 4>,
	pub targets: Tensor<B, 1>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LaserBatcher;

impl<B, T> Batcher<B, LoadedLaserItem<T>, LaserBatch<B>> for LaserBatcher
where
	B: Backend,
	T: NumericLike + Into<IntensityValType>,
{
	fn batch(
		&self,
		items: Vec<LoadedLaserItem<T>>,
		device: &Device<B>,
	) -> LaserBatch<B> {
		batch_items(items, device, identity)
	}
}

#[derive(Clone)]
pub struct AugmentedLaserBatcher<P>
where
	P: ImageStep,
{
	augmentation: P,
	rng: Arc<Mutex<StdRng>>,
}

impl<P> AugmentedLaserBatcher<P>
where
	P: ImageStep,
{
	/// Create a batcher with an augmentation pipeline defined at compile time.
	pub fn new(
		augmentation: P,
		seed: u64,
	) -> Self {
		Self {
			augmentation,
			rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
		}
	}
}

impl<B, T, P> Batcher<B, LoadedLaserItem<T>, LaserBatch<B>>
	for AugmentedLaserBatcher<P>
where
	B: Backend,
	T: NumericLike + Into<IntensityValType>,
	P: ImageStep,
{
	fn batch(
		&self,
		items: Vec<LoadedLaserItem<T>>,
		device: &Device<B>,
	) -> LaserBatch<B> {
		batch_items(items, device, |image| {
			let mut rng =
				self.rng.lock().unwrap_or_else(PoisonError::into_inner);
			self.augmentation.apply(image, &mut rng)
		})
	}
}

fn batch_items<B, T, F>(
	items: Vec<LoadedLaserItem<T>>,
	device: &Device<B>,
	mut modify_image: F,
) -> LaserBatch<B>
where
	B: Backend,
	T: NumericLike + Into<IntensityValType>,
	F: FnMut(ImageType) -> ImageType,
{
	let targets = items
		.iter()
		.map(|item| Tensor::<B, 1>::from_data([item.intensity.into()], device))
		.collect();
	let targets = Tensor::cat(targets, 0);

	let images = items.into_iter().map(|item| modify_image(item.image));
	let images = images_to_tensor(images, device);

	LaserBatch {
		images,
		targets,
	}
}

/// Convert from channel-last RGB to channel-first RGB.
///
/// Convolution layers are channel-first.
fn images_to_tensor<B, I>(
	images: I,
	device: &B::Device,
) -> Tensor<B, 4>
where
	B: Backend,
	I: ExactSizeIterator<Item = ImageType>,
{
	let batch_size = images.len();
	let image_size = IMG_PIXELS * 3;
	let mut values = vec![0; batch_size * image_size];

	for (image_values, image) in values.chunks_exact_mut(image_size).zip(images)
	{
		debug_assert_eq!(
			image.dimensions(),
			(IMG_WIDTH, IMG_HEIGHT),
			"The image did not have the correct dimensions"
		);

		let (red_values, other_channels) =
			image_values.split_at_mut(IMG_PIXELS);
		let (green_values, blue_values) =
			other_channels.split_at_mut(IMG_PIXELS);
		for (((red_slot, green_slot), blue_slot), pixel) in red_values
			.iter_mut()
			.zip(green_values)
			.zip(blue_values)
			.zip(image.pixels())
		{
			let image::Rgb([red, green, blue]) = *pixel;
			*red_slot = red;
			*green_slot = green;
			*blue_slot = blue;
		}
	}

	let data = TensorData::new(
		values,
		[batch_size, 3, IMG_HEIGHT as usize, IMG_WIDTH as usize],
	);
	Tensor::from_data(data, device)
}

#[cfg(test)]
mod tests {
	use burn_flex::{
		Flex,
		FlexDevice,
	};

	use super::*;
	const IMG_CHANNELS: u32 = 3;

	#[test]
	fn image_batch_tensor_is_nchw_and_keeps_images_separate() {
		let images = [
			ImageType::from_pixel(
				IMG_WIDTH,
				IMG_HEIGHT,
				image::Rgb([10, 20, 30]),
			),
			ImageType::from_pixel(
				IMG_WIDTH,
				IMG_HEIGHT,
				image::Rgb([40, 50, 60]),
			),
		];
		let device = FlexDevice::default();
		let tensor = images_to_tensor::<Flex, _>(images.into_iter(), &device);
		let dimensions = tensor.dims();
		let values = tensor
			.into_data()
			.to_vec::<f32>()
			.expect("tensor values should be readable");

		assert_eq!(
			dimensions,
			[
				2,
				IMG_CHANNELS as usize,
				IMG_HEIGHT as usize,
				IMG_WIDTH as usize
			]
		);
		assert_eq!(values.first(), Some(&10.0));
		assert_eq!(values.get(IMG_PIXELS), Some(&20.0));
		assert_eq!(values.get(2 * IMG_PIXELS), Some(&30.0));
		assert_eq!(values.get(3 * IMG_PIXELS), Some(&40.0));
		assert_eq!(values.get(4 * IMG_PIXELS), Some(&50.0));
		assert_eq!(values.get(5 * IMG_PIXELS), Some(&60.0));
	}
}
