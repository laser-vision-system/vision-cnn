//! Execute sample inference or full-split evaluation.

use std::error::Error;

use burn::{
	backend::{
		Cuda,
		cuda::CudaDevice,
	},
	prelude::Backend,
};
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

struct DatasetSplits<'a, B: Backend> {
	validation: &'a LoadedDataset<B>,
	test: &'a LoadedDataset<B>,
}

pub fn run(args: &InferArgs) -> Result<(), Box<dyn Error>> {
	type InferBackend = Cuda;
	let paths = paths::resolve(args)?;
	paths::prepare_reports(&paths, args.force)?;
	let device = CudaDevice::default();
	let splits = args.dataset.source().load_splits::<InferBackend>();
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

fn run_split<B>(
	arguments: &InferArgs,
	paths: &InferPaths,
	splits: &DatasetSplits<'_, B>,
	device: &B::Device,
) -> Result<(), Box<dyn Error>>
where
	B: Backend,
{
	if arguments.evaluate_all {
		evaluate_split(arguments, paths, splits, device)
	} else {
		evaluate::infer_samples::<B>(
			&paths.model_dir,
			&paths.model_name,
			arguments.num_samples.get(),
			splits.test,
			device,
		)
	}
}

fn evaluate_split<B>(
	arguments: &InferArgs,
	paths: &InferPaths,
	splits: &DatasetSplits<'_, B>,
	device: &B::Device,
) -> Result<(), Box<dyn Error>>
where
	B: Backend,
{
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
	evaluate::evaluate_all_named::<B>(
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
