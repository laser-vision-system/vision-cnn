//! Training command hierarchy and subcommand dispatch.

mod args;
mod engine;
mod new;
mod paths;
mod resume;
mod workflow;

use std::error::Error;

use clap::{
	Args,
	Subcommand,
};

#[derive(Args, Debug)]
/// Arguments accepted by the `train` command.
pub struct TrainArgs {
	#[command(subcommand)]
	command: TrainCommand,
}

#[derive(Debug, Subcommand)]
enum TrainCommand {
	#[command(about = "Train a new model")]
	New(new::TrainNewArgs),
	#[command(about = "Resume training from an existing model")]
	Resume(resume::TrainResumeArgs),
}

/// Execute the selected training mode.
///
/// # Errors
///
/// Returns an error when arguments, paths, training, evaluation, or export
/// fail.
pub fn run(args: TrainArgs) -> Result<(), Box<dyn Error>> {
	let request = match args.command {
		TrainCommand::New(new_args) => new_args.into_request(),
		TrainCommand::Resume(resume_args) => resume_args.into_request(),
	};
	workflow::run(&request)
}
