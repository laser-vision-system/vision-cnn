//! Execute sample inference or full-split evaluation.

use std::error::Error;

use burn::tensor::Device;
use vision_cnn::data::LoadedDataset;

use super::{
	args::{
		EvaluationSplit,
		InferArgs,
	},
	evaluate,
	paths::{
		self,
		InferPaths,
	},
};

struct DatasetSplits<'a> {
	validation: &'a LoadedDataset,
	test: &'a LoadedDataset,
}

pub fn run(args: &InferArgs) -> Result<(), Box<dyn Error>> {
	let paths = paths::resolve(args)?;
	paths::prepare_reports(&paths, args.force)?;
	let device = Device::default().autodiff();
	let splits = args.dataset.source().load_splits();
	run_split(
		args,
		&paths,
		&DatasetSplits {
			validation: &splits.validation,
			test: &splits.test,
		},
		&device,
	)
}

fn run_split(
	arguments: &InferArgs,
	paths: &InferPaths,
	splits: &DatasetSplits<'_>,
	device: &Device,
) -> Result<(), Box<dyn Error>> {
	if arguments.evaluate_all {
		evaluate_split(arguments, paths, splits, device)
	} else {
		evaluate::infer_samples(
			&paths.model_dir,
			&paths.model_name,
			arguments.num_samples.get(),
			splits.test,
			device,
		)
	}
}

fn evaluate_split(
	arguments: &InferArgs,
	paths: &InferPaths,
	splits: &DatasetSplits<'_>,
	device: &Device,
) -> Result<(), Box<dyn Error>> {
	let (split_name, dataset) = match arguments.evaluation_split {
		EvaluationSplit::Validate => ("validation", splits.validation),
		EvaluationSplit::Test => ("test", splits.test),
	};
	let report_dir = paths.report_dir.as_deref().ok_or_else(|| {
		std::io::Error::new(
			std::io::ErrorKind::InvalidInput,
			"evaluation report directory was not resolved",
		)
	})?;
	evaluate::evaluate_all_named(
		&evaluate::ModelExport {
			directory: &paths.model_dir,
			name: &paths.model_name,
		},
		arguments.evaluation_batch_size,
		arguments.dataset.name(),
		split_name,
		report_dir,
		dataset,
		device,
	)
}
