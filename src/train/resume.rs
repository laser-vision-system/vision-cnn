//! Arguments for continuing training from an existing model.

use std::path::PathBuf;

use clap::Args;

use super::{
	args::{
		TrainCommonArgs,
		TrainOverrideArgs,
	},
	workflow::TrainRequest,
};

#[derive(Args, Debug)]
pub struct TrainResumeArgs {
	/// Existing model directory name under the configured output base.
	#[arg(value_name = "MODEL_DIR")]
	source: PathBuf,
	/// Optional output directory name; defaults to the source directory.
	#[arg(short = 'o', long = "output", value_name = "OUTPUT_NAME")]
	output_name: Option<String>,
	#[command(flatten)]
	options: TrainCommonArgs,
	#[command(flatten)]
	overrides: TrainOverrideArgs,
}

impl TrainResumeArgs {
	pub fn into_request(self) -> TrainRequest {
		TrainRequest {
			source: Some(self.source),
			output_name: self.output_name,
			options: self.options,
			overrides: self.overrides,
		}
	}
}
