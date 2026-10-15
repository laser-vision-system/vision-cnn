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
	data::dataset::Dataset,
	tensor::Element,
};
use clap::{
	Args,
	ValueEnum,
};

use super::DatasetChoice;
use crate::data::{
	self,
	IntensityValType,
	LoadedLaserItem,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EvaluationSplit {
	Validate,
	Test,
}

#[derive(Args)]
pub(super) struct Arguments {
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

pub(super) fn run(arguments: &Arguments) -> Result<(), Box<dyn Error>> {
	let device = CudaDevice::default();
	match arguments.dataset {
		DatasetChoice::Analogous => {
			let (_, validate, test) = data::analogous_laser_splits();
			run_split(arguments, "analogous", &validate, &test, &device)
		},
		DatasetChoice::Synthetic1 => {
			let (_, validate, test) = data::synthetic1_splits();
			run_split(arguments, "synthetic1", &validate, &test, &device)
		},
		DatasetChoice::Synthetic2 => {
			let (_, validate, test) = data::synthetic2_splits();
			run_split(arguments, "synthetic2", &validate, &test, &device)
		},
		DatasetChoice::InLabSingle => {
			let (_, validate, test) = data::in_lab_single_laser_splits();
			run_split(arguments, "in-lab-single", &validate, &test, &device)
		},
	}
}

fn run_split<T, D>(
	arguments: &Arguments,
	dataset_name: &str,
	validation_split: &D,
	test_split: &D,
	device: &CudaDevice,
) -> Result<(), Box<dyn Error>>
where
	T: Element + Into<IntensityValType>,
	D: Dataset<LoadedLaserItem<T>>,
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
		evaluate::evaluate_all::<Cuda, T, D>(
			&arguments.model_dir,
			arguments.evaluation_batch_size,
			dataset_name,
			split_name,
			report_dir,
			dataset,
			device,
		)
	} else {
		evaluate::infer_samples::<Cuda, T, D>(
			&arguments.model_dir,
			arguments.num_samples.get(),
			test_split,
			device,
		)
	}
}
