//! Inference arguments and evaluation workflow.

mod evaluate;

use std::{
	error::Error,
	io::{
		Error as IoError,
		ErrorKind,
	},
	num::NonZeroUsize,
	path::PathBuf,
};

use burn::{
	backend::{
		Cuda,
		cuda::CudaDevice,
	},
	prelude::Backend,
};
use clap::{
	Args,
	ValueEnum,
};
use vision_cnn::data::LoadedDataset;

use super::DatasetChoice;

#[derive(Clone, Copy, Debug, ValueEnum)]
/// Evaluation split accepted by the `infer --evaluate-all` command.
enum EvaluationSplit {
	Validate,
	Test,
}

#[derive(Args)]
/// Arguments accepted by the `infer` command.
pub struct Arguments {
	#[arg(
		value_name = "MODEL_DIR",
		help = "Directory containing the trained model and config"
	)]
	model_dir: PathBuf,
	#[arg(
		short = 'd',
		long,
		value_name = "DATASET",
		value_enum,
		default_value = "in-lab-single",
		help = "Dataset to evaluate against"
	)]
	dataset: DatasetChoice,
	#[arg(
		short = 'n',
		long = "num-samples",
		value_name = "COUNT",
		default_value = "1",
		help = "Number of test samples to infer"
	)]
	num_samples: NonZeroUsize,
	#[arg(
		long = "evaluate-all",
		help = "Evaluate the complete test split and report aggregate metrics"
	)]
	evaluate_all: bool,
	#[arg(
		long,
		value_name = "COUNT",
		default_value = "32",
		requires = "evaluate_all",
		help = "Batch size used by --evaluate-all"
	)]
	evaluation_batch_size: NonZeroUsize,
	#[arg(
		long,
		value_enum,
		default_value = "test",
		requires = "evaluate_all",
		help = "Split used by --evaluate-all"
	)]
	evaluation_split: EvaluationSplit,
	#[arg(
		long,
		value_name = "DIR",
		requires = "evaluate_all",
		required_if_eq("evaluate_all", "true"),
		help = "Directory for metrics.json and samples.csv"
	)]
	report_dir: Option<PathBuf>,
}

pub fn run(arguments: &Arguments) -> Result<(), Box<dyn Error>> {
	type InferBackend = Cuda;
	let device = CudaDevice::default();
	let splits = arguments.dataset.source().load_splits::<InferBackend>();
	run_split::<InferBackend>(
		arguments,
		arguments.dataset.name(),
		&splits.validation,
		&splits.test,
		&device,
	)
}

/// Execute sample inference or full-split evaluation.
fn run_split<B>(
	arguments: &Arguments,
	dataset_name: &str,
	validation_split: &LoadedDataset<B>,
	test_split: &LoadedDataset<B>,
	device: &B::Device,
) -> Result<(), Box<dyn Error>>
where
	B: Backend,
{
	if arguments.evaluate_all {
		let (split_name, dataset) = match arguments.evaluation_split {
			EvaluationSplit::Validate => ("validation", validation_split),
			EvaluationSplit::Test => ("test", test_split),
		};
		let report_dir = arguments.report_dir.as_deref().ok_or_else(|| {
			IoError::new(
				ErrorKind::InvalidInput,
				"--report-dir is required with --evaluate-all",
			)
		})?;
		evaluate::evaluate_all::<B>(
			&arguments.model_dir,
			arguments.evaluation_batch_size,
			dataset_name,
			split_name,
			report_dir,
			dataset,
			device,
		)
	} else {
		evaluate::infer_samples::<B>(
			&arguments.model_dir,
			arguments.num_samples.get(),
			test_split,
			device,
		)
	}
}
