use std::{
	convert::identity,
	sync::{Arc, Mutex, PoisonError},
};

use burn::{data::dataloader::batcher::Batcher, prelude::*};
use dataset_loader::item::{ImageType, NumericLike};
use rand::{SeedableRng as _, rngs::StdRng};

use super::{ImageStep, IntensityValType};
use crate::data::LoadedLaserItem;

const IMG_CHANNELS: usize = 3;
const IMG_HEIGHT: usize = 480;
const IMG_WIDTH: usize = 640;

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
	/// Create a batcher with anaugmentation pipeline defined at compile time.
	pub fn new(augmentation: P, seed: u64) -> Self {
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
		.map(|item| {
			Tensor::<B, 1>::from_data(
				[item.intensity.into().elem::<B::FloatElem>()],
				device,
			)
		})
		.collect();

	let images = items
		.into_iter()
		.map(|item| {
			let image = modify_image(item.image);
			channel_last_to_channel_first(image)
		})
		.map(|data| Tensor::<B, 3>::from_data(data, device))
		.map(|tensor| tensor.unsqueeze_dim(0))
		.collect();

	let images = Tensor::cat(images, 0);
	let targets = Tensor::cat(targets, 0);

	LaserBatch { images, targets }
}

/// Convert from channel-last RGB to channel-first RGB.
///
/// Convolution layers are channel-first.
#[expect(
	clippy::needless_pass_by_value,
	reason = "This is a consuming function"
)]
fn channel_last_to_channel_first(image: ImageType) -> TensorData {
	debug_assert_eq!(
		image.width() as usize * image.height() as usize,
		IMG_HEIGHT * IMG_WIDTH,
		"mapped image dimensions must match the model input dimensions",
	);
	let pixel_count = IMG_HEIGHT * IMG_WIDTH;
	let (mut red_channel, mut green_channel, mut blue_channel) = (
		Vec::with_capacity(pixel_count),
		Vec::with_capacity(pixel_count),
		Vec::with_capacity(pixel_count),
	);
	for pixel in image.pixels() {
		let [red, green, blue] = pixel.0;
		red_channel.push(red);
		green_channel.push(green);
		blue_channel.push(blue);
	}
	let pixels = red_channel
		.into_iter()
		.chain(green_channel)
		.chain(blue_channel)
		.collect();
	TensorData::new(pixels, [IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH])
}
