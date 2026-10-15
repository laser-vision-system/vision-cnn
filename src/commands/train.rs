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

use super::DatasetChoice;
use crate::{
	data::{
		self,
		ImageStep as _,
		NoOp,
		augmentation::{
			Sequence,
			flip::{
				RandomHorizontalFlip,
				RandomVerticalFlip,
			},
			hue::HueRotate,
		},
	},
	training,
};

#[derive(Args)]
pub(super) struct Arguments {
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

pub(super) fn run(arguments: &Arguments) -> Result<(), Box<dyn Error>> {
	let plan = plan::load(arguments.resume_dir.as_deref())?;
	output::prepare(
		&arguments.output_dir,
		arguments.resume_dir.as_deref(),
		arguments.force,
	)?;
	run_training(&arguments.output_dir, &plan, arguments.dataset.source())
}

#[expect(clippy::expect_used, reason = "This should be a static assertion")]
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

	let pipeline = random_flip_augmentation_pipeline()
		.expect("The augmentation pipeline should use valid values");

	let train_pipeline = pipeline
		.clone()
		.then(HueRotate::new(-180, 180).expect("Should be valid range"));

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

fn random_flip_augmentation_pipeline()
-> Option<Sequence<RandomHorizontalFlip, RandomVerticalFlip>> {
	let pipeline =
		RandomHorizontalFlip::new(0.5)?.then(RandomVerticalFlip::new(0.5)?);
	Some(pipeline)
}
