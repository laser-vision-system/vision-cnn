//! Arguments for creating a new model.

use clap::Args;

use super::{
	args::{
		TrainCommonArgs,
		TrainOverrideArgs,
	},
	workflow::TrainRequest,
};

#[derive(Args, Debug)]
pub struct TrainNewArgs {
	/// Name of the model artefact directory under the configured output base.
	#[arg(value_name = "OUTPUT_NAME", required_unless_present = "dry_run")]
	output_name: Option<String>,
	#[command(flatten)]
	options: TrainCommonArgs,
	#[command(flatten)]
	overrides: TrainOverrideArgs,
}

impl TrainNewArgs {
	pub fn into_request(self) -> TrainRequest {
		TrainRequest {
			source: None,
			output_name: self.output_name,
			options: self.options,
			overrides: self.overrides,
		}
	}
}
