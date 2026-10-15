#![expect(clippy::expect_used, reason = "Datasets should always load")]

use std::num::NonZeroU8;

use dataset_loader::{
	Splitable,
	dataset::LaserDatasetSplit,
	item::LaserItem as _,
	sources::{
		analogous_laser,
		in_lab_single_laser,
		synthetic_1,
		synthetic_2,
	},
};

const TRAIN_WEIGHT: NonZeroU8 = match NonZeroU8::new(80) {
	Some(weight) => weight,
	None => NonZeroU8::MIN,
};
const VALIDATE_WEIGHT: u8 = 15;
const TEST_WEIGHT: u8 = 2;

pub mod augmentation;
pub mod batching;
pub mod mapping;

pub use augmentation::{
	ImageStep,
	NoOp,
};
pub use batching::{
	AugmentedLaserBatcher,
	LaserBatch,
	LaserBatcher,
};
pub use mapping::{
	LaserItemMapper,
	LoadedLaserItem,
};

type InternalDatasetSplit<I, T> =
	LaserDatasetSplit<LaserItemMapper, I, LoadedLaserItem<T>>;

type Synthetic1ValType = synthetic_1::IntensityValType;
type Synthetic2ValType = synthetic_2::IntensityValType;
type AnalogousLaserValType = analogous_laser::IntensityValType;
type InLabSingleLaserValType = in_lab_single_laser::IntensityValType;

type Synthetic1DatasetSplit =
	InternalDatasetSplit<synthetic_1::Synthetic1Item, Synthetic1ValType>;
type Synthetic2DatasetSplit =
	InternalDatasetSplit<synthetic_2::Synthetic2Item, Synthetic2ValType>;
type AnalogousDatasetSplit = InternalDatasetSplit<
	analogous_laser::AnalogousLaserItem,
	AnalogousLaserValType,
>;
type InLabSingleLaserDatasetSplit = InternalDatasetSplit<
	in_lab_single_laser::InLabSingleLaserItem,
	InLabSingleLaserValType,
>;

pub type IntensityValType = InLabSingleLaserValType;

fn split_dataset<D, MO>(
	dataset: D,
	train_weight: NonZeroU8,
	test_weight: u8,
) -> (D::Output, Option<D::Output>, Option<D::Output>)
where
	D: Splitable<MO>,
	MO: Send + Sync + Clone,
{
	let splits = dataset.into_split(train_weight, VALIDATE_WEIGHT, test_weight);
	(splits.train, splits.validate, splits.test)
}

fn split_train_validate_test<D, MO>(
	dataset: D,
	train_weight: NonZeroU8,
) -> (D::Output, D::Output, D::Output)
where
	D: Splitable<MO>,
	MO: Send + Sync + Clone,
{
	let (train, validate, test) =
		split_dataset(dataset, train_weight, TEST_WEIGHT);
	(
		train,
		validate.expect("validation split should exist"),
		test.expect("test split should exist"),
	)
}

fn split_train_validate<D, MO>(
	dataset: D,
	train_weight: NonZeroU8,
) -> (D::Output, D::Output)
where
	D: Splitable<MO>,
	MO: Send + Sync + Clone,
{
	let (train, validate, _) =
		split_dataset(dataset, train_weight, TEST_WEIGHT);
	(train, validate.expect("validation split should exist"))
}

/// A function to load the synthetic1 dataset into the train/validate/test
/// splits.
///
/// # Panics
///
/// If dataset could not be loaded as the path.
pub fn synthetic1_splits() -> (
	Synthetic1DatasetSplit,
	Synthetic1DatasetSplit,
	Synthetic1DatasetSplit,
) {
	let dataset = synthetic_1::get_dataset(
		LaserItemMapper,
		"./datasets/synthetic_1/",
		&[],
	)
	.expect("dataset should load");
	split_train_validate_test(dataset, TRAIN_WEIGHT)
}

/// Load the first synthetic dataset into training and validation splits.
///
/// # Panics
///
/// If dataset could not be loaded as the path.
pub fn synthetic1_train_val_splits()
-> (Synthetic1DatasetSplit, Synthetic1DatasetSplit) {
	let dataset = synthetic_1::get_dataset(
		LaserItemMapper,
		"./datasets/synthetic_1/",
		&[],
	)
	.expect("dataset should load");
	split_train_validate(dataset, TRAIN_WEIGHT)
}

/// A function to load the synthetic2 dataset into the train/validate/test
/// splits.
///
/// # Panics
///
/// If dataset could not be loaded as the path.
pub fn synthetic2_splits() -> (
	Synthetic2DatasetSplit,
	Synthetic2DatasetSplit,
	Synthetic2DatasetSplit,
) {
	let dataset = synthetic_2::get_dataset(
		LaserItemMapper,
		"./datasets/synthetic_2/",
		&[],
	)
	.expect("dataset should load");
	split_train_validate_test(dataset, TRAIN_WEIGHT)
}

/// Load the second synthetic dataset into training and validation splits.
///
/// # Panics
///
/// If dataset could not be loaded as the path.
pub fn synthetic2_train_val_splits()
-> (Synthetic2DatasetSplit, Synthetic2DatasetSplit) {
	let dataset = synthetic_2::get_dataset(
		LaserItemMapper,
		"./datasets/synthetic_2/",
		&[],
	)
	.expect("dataset should load");
	split_train_validate(dataset, TRAIN_WEIGHT)
}

/// A function to load the analogous laser dataset into the train/validate/test
/// splits.
///
/// # Panics
///
/// If dataset could not be loaded as the path.
pub fn analogous_laser_splits() -> (
	AnalogousDatasetSplit,
	AnalogousDatasetSplit,
	AnalogousDatasetSplit,
) {
	let dataset = analogous_laser::get_dataset(
		LaserItemMapper,
		"./datasets/analogous_laser/",
		&[],
	)
	.expect("dataset should load");
	split_train_validate_test(dataset, TRAIN_WEIGHT)
}

/// Load the analogous laser dataset into training and validation splits.
///
/// # Panics
///
/// If dataset could not be loaded as the path.
pub fn analogous_laser_train_val_splits()
-> (AnalogousDatasetSplit, AnalogousDatasetSplit) {
	let dataset = analogous_laser::get_dataset(
		LaserItemMapper,
		"./datasets/analogous_laser/",
		&[],
	)
	.expect("dataset should load");
	split_train_validate(dataset, TRAIN_WEIGHT)
}

/// A function to load the in lab single laser dataset into the
/// train/validate/test splits.
///
/// # Panics
///
/// If dataset could not be loaded as the path.
pub fn in_lab_single_laser_splits() -> (
	InLabSingleLaserDatasetSplit,
	InLabSingleLaserDatasetSplit,
	InLabSingleLaserDatasetSplit,
) {
	let dataset = in_lab_single_laser::get_dataset(
		LaserItemMapper,
		"./datasets/in_lab_single_laser/",
		&[Box::new(|item| {
			item.get_exposure_absolute() > 105_u16
				&& item
					.get_position()
					.is_some_and(|pos| !pos.contains("impure"))
		})],
	)
	.expect("dataset should load");
	split_train_validate_test(dataset, TRAIN_WEIGHT)
}

/// Load the in lab single laser dataset into training and validation splits.
///
/// # Panics
///
/// If dataset could not be loaded as the path.
pub fn in_lab_single_laser_train_val_splits()
-> (InLabSingleLaserDatasetSplit, InLabSingleLaserDatasetSplit) {
	let dataset = in_lab_single_laser::get_dataset(
		LaserItemMapper,
		"./datasets/in_lab_single_laser/",
		&[Box::new(|item| {
			item.get_exposure_absolute() > 105_u16
				&& item
					.get_position()
					.is_some_and(|pos| !pos.contains("impure"))
		})],
	)
	.expect("dataset should load");
	split_train_validate(dataset, TRAIN_WEIGHT)
}

#[cfg(test)]
mod tests {
	use burn::data::dataset::Dataset;
	use dataset_loader::{
		Split,
		SplitType,
		Splitable,
	};

	use super::*;

	#[derive(Clone)]
	struct DummyDataset {
		values: Vec<usize>,
	}

	struct DummySplit {
		values: Vec<usize>,
		kind: SplitType,
	}

	impl Dataset<usize> for DummyDataset {
		fn get(
			&self,
			index: usize,
		) -> Option<usize> {
			self.values.get(index).copied()
		}

		fn len(&self) -> usize {
			self.values.len()
		}
	}

	impl Splitable<usize> for DummyDataset {
		type DatasetType = Self;
		type Output = DummySplit;

		fn get_dataset(&self) -> &Self::DatasetType {
			self
		}

		fn create_split(
			&self,
			start: usize,
			end: usize,
			kind: SplitType,
		) -> Self::Output {
			DummySplit {
				values: self
					.values
					.iter()
					.skip(start)
					.take(end.saturating_sub(start))
					.copied()
					.collect(),
				kind,
			}
		}
	}

	impl Split for DummySplit {
		fn split_type(&self) -> SplitType {
			self.kind
		}
	}

	#[test]
	fn train_validation_helper_matches_full_split_validation_membership() {
		let dataset = DummyDataset {
			values: (0..100).collect(),
		};
		let (_, validation_with_test, _) =
			split_train_validate_test(dataset.clone(), TRAIN_WEIGHT);
		let (_, validation_for_training) =
			split_train_validate(dataset, TRAIN_WEIGHT);

		assert_eq!(validation_for_training.values, validation_with_test.values);
		assert!(matches!(
			validation_for_training.split_type(),
			SplitType::Validate
		));
	}
}
