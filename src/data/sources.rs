//! Source selection and shared train/validation/test splitting.

#![expect(clippy::expect_used, reason = "Configured datasets must load")]

use burn::{
	data::dataset::Dataset,
	prelude::Backend,
};
use dataset_loader::{
	PositiveFinite,
	SplitWeights,
	Splitable,
	item::LaserItem as _,
	sources::{
		analogous_laser,
		in_lab_single_laser,
		synthetic_1,
		synthetic_2,
	},
};

use super::{
	LaserItemMapper,
	LoadedDataset,
	LoadedLaserItem,
};

const TRAIN_WEIGHT: PositiveFinite = match PositiveFinite::new(80.0) {
	Ok(weight) => weight,
	Err(_) => panic!("traning weight is invalid"),
};
const VALIDATE_WEIGHT: PositiveFinite = match PositiveFinite::new(15.0) {
	Ok(weight) => weight,
	Err(_) => panic!("validation weight is invalid"),
};
const TEST_WEIGHT: PositiveFinite = match PositiveFinite::new(2.0) {
	Ok(weight) => weight,
	Err(_) => panic!("testing weight is invalid"),
};

const WEIGHTS: SplitWeights =
	match SplitWeights::try_new(TRAIN_WEIGHT, VALIDATE_WEIGHT, TEST_WEIGHT) {
		Some(value) => value,
		None => panic!("Split weights are invalid"),
	};

/// A supported dataset source.
#[derive(Clone, Copy, Debug)]
pub enum DatasetSource {
	Synthetic1,
	Synthetic2,
	Analogous,
	InLabSingle,
}

impl DatasetSource {
	/// Load the source and return its train, validation, and test splits.
	///
	/// # Panics
	///
	/// Panics if the selected dataset cannot be loaded or either evaluation
	/// split is unavailable.
	pub fn load_splits<B>(self) -> DatasetSplits<B>
	where
		B: Backend,
	{
		match self {
			Self::Synthetic1 => {
				split_source(
					synthetic_1::get_dataset(
						LaserItemMapper::<B>::default(),
						"./datasets/synthetic_1/",
						&[],
					)
					.expect("synthetic_1 dataset should load"),
				)
			},
			Self::Synthetic2 => {
				split_source(
					synthetic_2::get_dataset(
						LaserItemMapper::<B>::default(),
						"./datasets/synthetic_2/",
						&[],
					)
					.expect("synthetic_2 dataset should load"),
				)
			},
			Self::Analogous => {
				split_source(
					analogous_laser::get_dataset(
						LaserItemMapper::<B>::default(),
						"./datasets/analogous_laser/",
						&[],
					)
					.expect("analogous_laser dataset should load"),
				)
			},
			Self::InLabSingle => {
				split_source(
					in_lab_single_laser::get_dataset(
						LaserItemMapper::<B>::default(),
						"./datasets/in_lab_single_laser/",
						&[Box::new(|item| {
							item.get_exposure_absolute() > 105_u16
								&& item
									.get_position()
									.is_some_and(|pos| !pos.contains("impure"))
						})],
					)
					.expect("in_lab_single_laser dataset should load"),
				)
			},
		}
	}
}

/// The three mapped splits from a selected dataset.
pub struct DatasetSplits<B: Backend> {
	/// Samples used to update the model.
	pub training: LoadedDataset<B>,
	/// Samples used to select model checkpoints.
	pub validation: LoadedDataset<B>,
	/// Samples reserved for final evaluation.
	pub test: LoadedDataset<B>,
}

fn split_source<B, D>(dataset: D) -> DatasetSplits<B>
where
	B: Backend,
	D: Splitable<LoadedLaserItem<B>>,
	D::Output: Dataset<LoadedLaserItem<B>> + 'static,
{
	let splits = dataset.into_split(WEIGHTS);
	DatasetSplits {
		training: Box::new(splits.train.expect("training split should exist")),
		validation: Box::new(
			splits.validate.expect("validation split should exist"),
		),
		test: Box::new(splits.test.expect("test split should exist")),
	}
}
