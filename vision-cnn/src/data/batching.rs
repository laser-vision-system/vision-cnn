//! Image batching operations for training, validation, and testing.

#![expect(
	clippy::allow_attributes,
	reason = "Expect may not be fulfilled with swappable batching functions"
)]

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
use dataset_loader::augmentation::ImageStep;
use image::{
	GrayImage,
	RgbImage,
};
use rand::{
	SeedableRng as _,
	rngs::StdRng,
};

use super::{
	IMG_CHANNELS_LUMA,
	IMG_CHANNELS_RGB,
	IMG_HEIGHT,
	IMG_PIXELS,
	IMG_WIDTH,
	ImageType,
	LoadedLaserItem,
};

const BATCH_FEATURE_RANK: usize = 4;
const BATCH_LABEL_RANK: usize = 1;

/// A collection of input items and labels.
#[derive(Clone, Debug)]
pub struct LaserBatch<B: Backend> {
	/// The input features.
	pub images: Tensor<B, BATCH_FEATURE_RANK>,
	/// The labels.
	pub targets: Tensor<B, BATCH_LABEL_RANK>,
}

/// A unit struct that provides the batching trait implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaserBatcher;

// Has two different backend generics because of autodiff.
impl<B, M> Batcher<B, LoadedLaserItem<M>, LaserBatch<B>> for LaserBatcher
where
	B: Backend,
	M: Backend<FloatElem = B::FloatElem>,
{
	fn batch(
		&self,
		items: Vec<LoadedLaserItem<M>>,
		device: &Device<B>,
	) -> LaserBatch<B> {
		batch_items(items, device, identity)
	}
}

/// Provides image augmentation alongside batching operations.
#[derive(Clone)]
pub struct AugmentedLaserBatcher<P: ImageStep> {
	augmentation: P,
	rng: Arc<Mutex<StdRng>>,
}

impl<P: ImageStep> AugmentedLaserBatcher<P> {
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

// Has two different backend generics because of autodiff.
impl<B, M, P> Batcher<B, LoadedLaserItem<M>, LaserBatch<B>>
	for AugmentedLaserBatcher<P>
where
	B: Backend,
	M: Backend<FloatElem = B::FloatElem>,
	P: ImageStep,
{
	fn batch(
		&self,
		items: Vec<LoadedLaserItem<M>>,
		device: &Device<B>,
	) -> LaserBatch<B> {
		batch_items(items, device, |image| {
			let mut rng =
				self.rng.lock().unwrap_or_else(PoisonError::into_inner);
			self.augmentation.apply(image, &mut rng)
		})
	}
}

// Has two different backend generics because of autodiff.
fn batch_items<B, M, F>(
	items: Vec<LoadedLaserItem<M>>,
	device: &Device<B>,
	mut modify_image: F,
) -> LaserBatch<B>
where
	B: Backend,
	M: Backend<FloatElem = B::FloatElem>,
	F: FnMut(ImageType) -> ImageType,
{
	let targets = items
		.iter()
		.map(|item| {
			Tensor::<B, BATCH_LABEL_RANK>::from_data([item.intensity], device)
		})
		.collect();
	let targets = Tensor::cat(targets, 0);

	let images = items.into_iter().map(|item| modify_image(item.image));
	let images = luma_images_to_tensor(images, device);

	LaserBatch {
		images,
		targets,
	}
}

/// Batch luma images into a tensor.
#[allow(dead_code, reason = "Batching options can be swapped when needed")]
fn luma_images_to_tensor<B, I>(
	images: I,
	device: &B::Device,
) -> Tensor<B, BATCH_FEATURE_RANK>
where
	B: Backend,
	I: ExactSizeIterator<Item = GrayImage>,
{
	let batch_size = images.len();

	let values = images
		.inspect(|img| {
			debug_assert_eq!(
				img.dimensions(),
				(IMG_WIDTH, IMG_HEIGHT),
				"The image did not have the correct dimensions"
			);
		})
		.flat_map(|img| img.to_vec())
		.collect();

	let data = TensorData::new(
		values,
		[
			batch_size,
			IMG_CHANNELS_LUMA,
			IMG_HEIGHT as usize,
			IMG_WIDTH as usize,
		],
	);
	Tensor::from_data(data, device)
}

/// Convert from channel-last RGB to channel-first RGB.
///
/// Convolution layers are channel-first.
#[allow(dead_code, reason = "Batching options can be swapped when needed")]
fn rgb_images_to_tensor<B, I>(
	images: I,
	device: &B::Device,
) -> Tensor<B, BATCH_FEATURE_RANK>
where
	B: Backend,
	I: ExactSizeIterator<Item = RgbImage>,
{
	let batch_size = images.len();
	let image_size = IMG_PIXELS * IMG_CHANNELS_RGB;
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
		[
			batch_size,
			IMG_CHANNELS_RGB,
			IMG_HEIGHT as usize,
			IMG_WIDTH as usize,
		],
	);
	Tensor::from_data(data, device)
}

#[cfg(test)]
mod tests {
	use burn_flex::{
		Flex,
		FlexDevice,
	};
	use image::RgbImage;

	use super::*;

	#[test]
	fn luma_image_batch_tensor_is_nchw_and_keeps_images_separate() {
		let images = [
			GrayImage::from_pixel(IMG_WIDTH, IMG_HEIGHT, image::Luma([10])),
			GrayImage::from_pixel(IMG_WIDTH, IMG_HEIGHT, image::Luma([20])),
		];
		let batch_size = images.len();
		let device = FlexDevice;
		let tensor =
			luma_images_to_tensor::<Flex, _>(images.into_iter(), &device);
		let dimensions = tensor.dims();
		let values = tensor
			.into_data()
			.to_vec::<f32>()
			.expect("tensor values should be readable");

		assert_eq!(
			dimensions,
			[
				batch_size,
				IMG_CHANNELS_LUMA,
				IMG_HEIGHT as usize,
				IMG_WIDTH as usize
			]
		);
		assert_eq!(values.first(), Some(&10.0));
		assert_eq!(values.get(IMG_PIXELS), Some(&20.0));
	}

	#[test]
	fn rgb_image_batch_tensor_is_nchw_and_keeps_images_separate() {
		let images = [
			RgbImage::from_pixel(
				IMG_WIDTH,
				IMG_HEIGHT,
				image::Rgb([10, 20, 30]),
			),
			RgbImage::from_pixel(
				IMG_WIDTH,
				IMG_HEIGHT,
				image::Rgb([40, 50, 60]),
			),
		];
		let batch_size = images.len();
		let device = FlexDevice;
		let tensor =
			rgb_images_to_tensor::<Flex, _>(images.into_iter(), &device);
		let dimensions = tensor.dims();
		let values = tensor
			.into_data()
			.to_vec::<f32>()
			.expect("tensor values should be readable");

		assert_eq!(
			dimensions,
			[
				batch_size,
				IMG_CHANNELS_RGB,
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
