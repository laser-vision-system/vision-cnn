//! Build the training configuration and optional resume state for the CLI.

use std::{
	error::Error,
	io::{
		Error as IoError,
		ErrorKind,
	},
	path::Path,
};

use burn::{
	optim::AdamConfig,
	prelude::Config as _,
};

use crate::{
	model::LaserModelConfig,
	training::{
		TrainingConfig,
		latest_checkpoint_epoch,
	},
};

/// Resolved training configuration and optional checkpoint to resume from.
pub struct TrainingPlan {
	config: TrainingConfig,
	resume: Option<(String, usize)>,
}

/// Optional command-line and environment overrides for training settings.
#[derive(Default)]
pub struct TrainingOverrides {
	/// Number of epochs to run in this training invocation.
	pub num_epochs: Option<usize>,
	/// Number of samples per batch.
	pub batch_size: Option<usize>,
	/// Number of data-loading workers.
	pub num_workers: Option<usize>,
	/// Random seed.
	pub seed: Option<u64>,
	/// Initial learning rate.
	pub learning_rate: Option<f64>,
	/// Inverse-time learning-rate decay factor.
	pub decay_rate: Option<f64>,
	/// Learning-rate decay interval.
	pub decay_steps: Option<usize>,
	/// Validation epochs without improvement before stopping.
	pub early_stopping_patience: Option<usize>,
}

impl TrainingPlan {
	/// Return the configuration used for this training run.
	pub const fn config(&self) -> &TrainingConfig {
		&self.config
	}

	/// Return the checkpoint source and epoch, when resuming.
	pub fn resume(&self) -> Option<(&str, usize)> {
		self.resume
			.as_ref()
			.map(|(directory, epoch)| (directory.as_str(), *epoch))
	}
}

/// Load a new or resumed training plan.
///
/// # Errors
///
/// Returns an error when a resume configuration or checkpoint cannot be read.
pub fn load(
	resume_dir: Option<&Path>,
	overrides: &TrainingOverrides,
) -> Result<TrainingPlan, Box<dyn Error>> {
	let mut plan = match resume_dir {
		Some(directory) => load_resume_plan(directory)?,
		None => new_training_plan(),
	};
	plan.config = apply_overrides(plan.config, overrides);
	Ok(plan)
}

fn apply_overrides(
	config: TrainingConfig,
	overrides: &TrainingOverrides,
) -> TrainingConfig {
	let num_epochs = overrides.num_epochs.unwrap_or(config.num_epochs);
	let batch_size = overrides.batch_size.unwrap_or(config.batch_size);
	let num_workers = overrides.num_workers.unwrap_or(config.num_workers);
	let seed = overrides.seed.unwrap_or(config.seed);
	let learning_rate = overrides.learning_rate.unwrap_or(config.learning_rate);
	let decay_rate = overrides.decay_rate.unwrap_or(config.decay_rate);
	let decay_steps = overrides.decay_steps.unwrap_or(config.decay_steps);
	let early_stopping_patience = overrides
		.early_stopping_patience
		.unwrap_or(config.early_stopping_patience);

	config
		.with_num_epochs(num_epochs)
		.with_batch_size(batch_size)
		.with_num_workers(num_workers)
		.with_seed(seed)
		.with_learning_rate(learning_rate)
		.with_decay_rate(decay_rate)
		.with_decay_steps(decay_steps)
		.with_early_stopping_patience(early_stopping_patience)
}

fn new_training_plan() -> TrainingPlan {
	let model_config = LaserModelConfig::new();
	let training_config = TrainingConfig::new(model_config, AdamConfig::new());
	TrainingPlan {
		config: training_config,
		resume: None,
	}
}

fn load_resume_plan(directory: &Path) -> Result<TrainingPlan, Box<dyn Error>> {
	let config = TrainingConfig::load(directory.join("config.json"))?;
	let directory_text = directory.to_str().ok_or_else(|| {
		IoError::new(ErrorKind::InvalidInput, "resume path is not UTF-8")
	})?;
	let checkpoint =
		latest_checkpoint_epoch(directory_text).ok_or_else(|| {
			IoError::new(
				ErrorKind::NotFound,
				"no training checkpoint found in resume directory",
			)
		})?;

	Ok(TrainingPlan {
		config,
		resume: Some((directory_text.to_owned(), checkpoint)),
	})
}

#[cfg(test)]
mod tests {
	use std::{
		env,
		fs,
		path::PathBuf,
		process,
		sync::atomic::{
			AtomicUsize,
			Ordering,
		},
	};

	use super::*;
	use crate::training::NUM_WORKERS_DEFAULT;

	static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

	fn create_temp_dir() -> PathBuf {
		let directory = env::temp_dir().join(format!(
			"cnn-training-plan-{}-{}",
			process::id(),
			NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed),
		));
		fs::create_dir_all(&directory)
			.expect("temporary plan directory should be created");
		directory
	}

	#[test]
	fn overrides_take_precedence_and_preserve_unmodified_resume_values() {
		let config = new_training_plan().config.with_seed(7).with_batch_size(3);
		let overrides = TrainingOverrides {
			seed: Some(18),
			..TrainingOverrides::default()
		};

		let resolved = apply_overrides(config, &overrides);

		assert_eq!(resolved.seed, 18);
		assert_eq!(resolved.batch_size, 3);
		assert_eq!(resolved.num_workers, NUM_WORKERS_DEFAULT);
	}

	#[test]
	fn resumed_num_epochs_are_additional_to_the_checkpoint_epoch() {
		let directory = create_temp_dir();
		let config = new_training_plan().config.with_num_epochs(5);
		config
			.save(directory.join("config.json"))
			.expect("training config should be saved");
		let checkpoint = directory.join("checkpoint");
		fs::create_dir_all(&checkpoint)
			.expect("checkpoint directory should be created");
		fs::write(checkpoint.join("model-5.mpk.gz"), [])
			.expect("checkpoint marker should be written");

		let overrides = TrainingOverrides {
			num_epochs: Some(2),
			..TrainingOverrides::default()
		};
		let plan = load(Some(&directory), &overrides)
			.expect("additional epochs should be accepted");
		assert_eq!(plan.config().num_epochs, 2);
		assert_eq!(plan.resume().map(|(_, epoch)| epoch), Some(5));
		fs::remove_dir_all(directory)
			.expect("temporary plan directory should be removed");
	}
}
