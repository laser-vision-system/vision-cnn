use std::{
	error::Error,
	io::{Error as IoError, ErrorKind},
	path::Path,
};

use burn::{optim::AdamConfig, prelude::Config as _};

use crate::{
	model::LaserModelConfig,
	training::{TrainingConfig, latest_checkpoint_epoch},
};

pub(super) struct TrainingPlan {
	config: TrainingConfig,
	resume: Option<(String, usize)>,
}

impl TrainingPlan {
	pub(super) const fn config(&self) -> &TrainingConfig {
		&self.config
	}

	pub(super) fn resume(&self) -> Option<(&str, usize)> {
		self.resume
			.as_ref()
			.map(|(directory, epoch)| (directory.as_str(), *epoch))
	}
}

pub(super) fn load(
	resume_dir: Option<&Path>,
) -> Result<TrainingPlan, Box<dyn Error>> {
	match resume_dir {
		Some(directory) => load_resume_plan(directory),
		None => Ok(new_training_plan()),
	}
}

fn new_training_plan() -> TrainingPlan {
	let model_config = LaserModelConfig::new().with_dropout(0.1);
	TrainingPlan {
		config: TrainingConfig::new(model_config, AdamConfig::new())
			.with_num_workers(8),
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
