//! Command workflows used by the CNN application.

mod inference;
mod training;

use std::error::Error;

use clap::{
	Parser,
	Subcommand,
	ValueEnum,
};

#[derive(Clone, Copy, ValueEnum)]
pub(super) enum DatasetChoice {
	#[value(name = "synthetic1")]
	Synthetic1,
	#[value(name = "synthetic2")]
	Synthetic2,
	Analogous,
	InLabSingle,
}

#[derive(Parser)]
#[command(
	name = "cnn",
	about = "Train and evaluate laser intensity models",
	subcommand_required = true,
	arg_required_else_help = true
)]
struct Cli {
	#[command(subcommand)]
	command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
	#[command(about = "Train a laser intensity model")]
	Train(training::Arguments),
	#[command(
		about = "Run inference or evaluate a trained laser intensity model"
	)]
	Infer(inference::Arguments),
}

pub fn run() -> Result<(), Box<dyn Error>> {
	match Cli::parse().command {
		CliCommand::Train(arguments) => training::run(&arguments),
		CliCommand::Infer(arguments) => inference::run(&arguments),
	}
}
