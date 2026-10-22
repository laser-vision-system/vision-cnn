//! Arguments for model inference and full-split evaluation.

use std::path::PathBuf;

use clap::{
	Args,
	ValueEnum,
};

use crate::DatasetChoice;

#[derive(Clone, Copy, Debug, ValueEnum)]
/// Evaluation split accepted by `infer --evaluate-all`.
pub enum EvaluationSplit {
	Validate,
	Test,
}

#[derive(Args, Debug)]
/// Arguments accepted by the `infer` command.
pub struct InferArgs {
	/// Model directory name under the configured output directory.
	#[arg(value_name = "MODEL_DIR")]
	pub model_dir: PathBuf,
	/// Exported model filename without its `.bpk` extension.
	#[arg(
		long = "name",
		visible_alias = "model-name",
		value_name = "MODEL_NAME"
	)]
	pub model_name: Option<String>,
	#[arg(
		short = 'd',
		long,
		value_name = "DATASET",
		value_enum,
		default_value = "in-lab-single",
		help = "Dataset to evaluate against"
	)]
	pub dataset: DatasetChoice,
	#[arg(short = 'n', long = "num-samples", default_value = "1")]
	pub num_samples: std::num::NonZeroUsize,
	/// Evaluate the complete selected split and report aggregate metrics.
	#[arg(long = "evaluate-all")]
	pub evaluate_all: bool,
	#[arg(long, default_value = "32", requires = "evaluate_all")]
	pub evaluation_batch_size: std::num::NonZeroUsize,
	#[arg(long, value_enum, default_value = "test", requires = "evaluate_all")]
	pub evaluation_split: EvaluationSplit,
	#[arg(long, value_name = "DIR", requires = "evaluate_all")]
	pub report_dir: Option<PathBuf>,
	/// Replace an existing report directory.
	#[arg(short = 'f', long)]
	pub force: bool,
}
