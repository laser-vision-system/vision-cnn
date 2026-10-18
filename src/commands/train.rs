//! Training command arguments and orchestration.

mod output;
mod plan;

use std::{
	error::Error,
	io::{
		Error as IoError,
		ErrorKind,
	},
	path::{
		Path,
		PathBuf,
	},
};

use burn::backend::{
	Autodiff,
	Cuda,
	cuda::CudaDevice,
};
use clap::Args;
use dataset_loader::augmentation::{
	ImageStep as _,
	NoOp,
	Sequence,
	flip::{
		FlipProbability,
		RandomHorizontalFlip,
		RandomVerticalFlip,
	},
	hue::{
		HueRange,
		HueRotate,
	},
};
use vision_cnn::{
	data,
	training,
};

use super::DatasetChoice;

#[derive(Args)]
/// Arguments accepted by the `train` command.
pub struct Arguments {
	#[arg(
		value_name = "OUTPUT_DIR",
		help = "Directory for the trained model and artifacts"
	)]
	output_dir: PathBuf,
	#[arg(
		short = 'r',
		long = "resume",
		value_name = "SOURCE_DIR",
		help = "Resume from this artifact directory"
	)]
	resume_dir: Option<PathBuf>,
	#[arg(
		short = 'd',
		long,
		value_name = "DATASET",
		value_enum,
		default_value = "in-lab-single",
		help = "Dataset to train against"
	)]
	dataset: DatasetChoice,
	#[arg(
		short = 'f',
		long,
		help = "Remove the output directory first if it exists"
	)]
	force: bool,
}

pub fn run(arguments: &Arguments) -> Result<(), Box<dyn Error>> {
	let plan = plan::load(arguments.resume_dir.as_deref())?;
	output::prepare(
		&arguments.output_dir,
		arguments.resume_dir.as_deref(),
		arguments.force,
	)?;
	run_training(&arguments.output_dir, &plan, arguments.dataset.source())
}

/// Run training using the selected artifact directory and dataset source.
fn run_training(
	output_dir: &Path,
	plan: &plan::TrainingPlan,
	source: data::DatasetSource,
) -> Result<(), Box<dyn Error>> {
	type TrainBackend = Cuda;
	type TrainAutodiffBackend = Autodiff<TrainBackend>;
	let device = CudaDevice::default();
	let splits = source.load_splits::<TrainBackend>();

	let output_dir = output_dir.to_str().ok_or_else(|| {
		IoError::new(ErrorKind::InvalidInput, "output path is not UTF-8")
	})?;

	let pipeline = random_flip_augmentation_pipeline();

	let train_pipeline = pipeline
		.clone()
		.then(HueRotate::new(HueRange::relaxed(-180, 180)));

	training::train::<TrainAutodiffBackend>(
		output_dir,
		plan.config(),
		&device,
		splits,
		train_pipeline,
		NoOp,
		plan.resume(),
	);
	Ok(())
}

/// Compose the random image-flip augmentations used during training.
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
