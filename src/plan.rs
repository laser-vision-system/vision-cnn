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
		BATCH_SIZE_DEFAULT,
		DECAY_RATE_DEFAULT,
		DECAY_STEPS_DEFAULT,
		EARLY_STOPPING_PATIENCE_DEFAULT,
		LEARNING_RATE_DEFAULT,
		NUM_EPOCHS_DEFAULT,
		NUM_WORKERS_DEFAULT,
		SEED_DEFAULT,
		TrainingConfig,
		latest_checkpoint_epoch,
	},
};

/// Resolved training configuration and optional checkpoint to resume from.
pub struct TrainingPlan {
	config: TrainingConfig,
	resume: Option<(String, usize)>,
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
pub fn load(resume_dir: Option<&Path>) -> Result<TrainingPlan, Box<dyn Error>> {
	match resume_dir {
		Some(directory) => load_resume_plan(directory),
		None => Ok(new_training_plan()),
	}
}

fn new_training_plan() -> TrainingPlan {
	let model_config = LaserModelConfig::new();
	// Set up like this to allow for command line arguments to modify.
	let training_config = TrainingConfig::new(model_config, AdamConfig::new())
		.with_num_epochs(NUM_EPOCHS_DEFAULT)
		.with_batch_size(BATCH_SIZE_DEFAULT)
		.with_num_workers(NUM_WORKERS_DEFAULT)
		.with_seed(SEED_DEFAULT)
		.with_learning_rate(LEARNING_RATE_DEFAULT)
		.with_decay_rate(DECAY_RATE_DEFAULT)
		.with_decay_steps(DECAY_STEPS_DEFAULT)
		.with_early_stopping_patience(EARLY_STOPPING_PATIENCE_DEFAULT);
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
