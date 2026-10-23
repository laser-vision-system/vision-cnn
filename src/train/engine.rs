//! Execute training and optional in-memory evaluation.

use std::{
	error::Error,
	io::{
		Error as IoError,
		ErrorKind,
	},
	path::Path,
};

use burn::backend::{
	Autodiff,
	Cuda,
	cuda::CudaDevice,
};
use dataset_loader::augmentation::{
	ImageStep as _,
	NoOp,
	Sequence,
	flip::{
		FlipProbability,
		RandomHorizontalFlip,
		RandomVerticalFlip,
	},
};
use vision_cnn::{
	data,
	model::LaserModel,
	plan::TrainingPlan,
	training,
};

use crate::DatasetChoice;

pub fn train(
	output_dir: &Path,
	plan: &TrainingPlan,
	source: data::DatasetSource,
) -> Result<LaserModel<Cuda>, Box<dyn Error>> {
	type TrainAutodiffBackend = Autodiff<Cuda>;
	let device = CudaDevice::default();
	let splits = source.load_splits::<Cuda>();
	let output_dir = output_dir.to_str().ok_or_else(|| {
		IoError::new(ErrorKind::InvalidInput, "output path is not UTF-8")
	})?;
	let train_pipeline = random_flip_augmentation_pipeline();
	Ok(training::train::<TrainAutodiffBackend>(
		output_dir,
		plan,
		&device,
		splits,
		train_pipeline,
		NoOp,
	))
}

pub fn evaluate(
	model: &LaserModel<Cuda>,
	report_dir: &Path,
	dataset_choice: DatasetChoice,
) -> Result<(), Box<dyn Error>> {
	let device = CudaDevice::default();
	let splits = dataset_choice.source().load_splits::<Cuda>();
	let batch_size = std::num::NonZeroUsize::new(32).ok_or_else(|| {
		IoError::new(ErrorKind::InvalidInput, "invalid evaluation batch size")
	})?;
	crate::infer::evaluate::evaluate_all_with_model::<Cuda>(
		model,
		batch_size,
		dataset_choice.name(),
		"test",
		report_dir,
		&splits.test,
		&device,
	)
}

fn random_flip_augmentation_pipeline()
-> Sequence<RandomHorizontalFlip, RandomVerticalFlip> {
	const FLIP_PROBABILITY: FlipProbability =
		match FlipProbability::try_new(0.5) {
			Ok(probability) => probability,
			Err(_) => panic!("invalid constant flip probability"),
		};
	RandomHorizontalFlip::new(FLIP_PROBABILITY)
		.then(RandomVerticalFlip::new(FLIP_PROBABILITY))
}
