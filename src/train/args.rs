//! Shared arguments for creating and resuming training runs.

use clap::Args;
use vision_cnn::plan::TrainingOverrides;

use crate::DatasetChoice;

#[derive(Clone, Args, Debug)]
pub struct TrainCommonArgs {
	#[arg(
		short = 'd',
		long,
		value_name = "DATASET",
		value_enum,
		default_value = "in-lab-single",
		help = "Dataset to train against"
	)]
	pub dataset: DatasetChoice,
	#[arg(short = 'f', long, help = "Replace an existing output directory")]
	pub force: bool,
	/// Optional custom exported model filename without its `.bpk` extension.
	#[arg(
		long = "name",
		value_name = "MODEL_NAME",
		help = "Custom name for the exported model file (no extension)"
	)]
	pub model_name: Option<String>,
	/// Evaluate the full test split after training.
	#[arg(long)]
	pub evaluate: bool,
	/// Train without exporting the model.
	#[arg(
		long,
		help = "Don't export the model; store artefacts in a timestamped temp folder"
	)]
	pub dry_run: bool,
}

#[derive(Clone, Args, Debug)]
pub struct TrainOverrideArgs {
	/// Number of epochs to run in this invocation.
	#[arg(long, value_name = "COUNT", env = "VISION_CNN_NUM_EPOCHS")]
	pub num_epochs: Option<usize>,
	/// Number of samples per batch.
	#[arg(long, value_name = "COUNT", env = "VISION_CNN_BATCH_SIZE")]
	pub batch_size: Option<usize>,
	/// Number of data-loading workers.
	#[arg(long, value_name = "COUNT", env = "VISION_CNN_NUM_WORKERS")]
	pub num_workers: Option<usize>,
	/// Random seed.
	#[arg(long, value_name = "SEED", env = "VISION_CNN_SEED")]
	pub seed: Option<u64>,
	/// Initial learning rate.
	#[arg(long, value_name = "RATE", env = "VISION_CNN_LEARNING_RATE")]
	pub learning_rate: Option<f64>,
	/// Inverse-time learning-rate decay factor.
	#[arg(long, value_name = "RATE", env = "VISION_CNN_DECAY_RATE")]
	pub decay_rate: Option<f64>,
	/// Learning-rate decay interval.
	#[arg(long, value_name = "COUNT", env = "VISION_CNN_DECAY_STEPS")]
	pub decay_steps: Option<usize>,
	/// Validation epochs without improvement before stopping.
	#[arg(
		long,
		value_name = "COUNT",
		env = "VISION_CNN_EARLY_STOPPING_PATIENCE"
	)]
	pub early_stopping_patience: Option<usize>,
}

impl From<TrainOverrideArgs> for TrainingOverrides {
	fn from(value: TrainOverrideArgs) -> Self {
		TrainingOverrides {
			num_epochs: value.num_epochs,
			batch_size: value.batch_size,
			num_workers: value.num_workers,
			seed: value.seed,
			learning_rate: value.learning_rate,
			decay_rate: value.decay_rate,
			decay_steps: value.decay_steps,
			early_stopping_patience: value.early_stopping_patience,
		}
	}
}
