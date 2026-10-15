//! Command workflows used by the CNN application.

mod infer;
mod train;

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
	#[value(name = "in-lab-single")]
	InLabSingle,
}

impl DatasetChoice {
	pub(super) const fn source(self) -> crate::data::DatasetSource {
		match self {
			Self::Synthetic1 => crate::data::DatasetSource::Synthetic1,
			Self::Synthetic2 => crate::data::DatasetSource::Synthetic2,
			Self::Analogous => crate::data::DatasetSource::Analogous,
			Self::InLabSingle => crate::data::DatasetSource::InLabSingle,
		}
	}

	pub(super) const fn name(self) -> &'static str {
		match self {
			Self::Synthetic1 => "synthetic1",
			Self::Synthetic2 => "synthetic2",
			Self::Analogous => "analogous",
			Self::InLabSingle => "in-lab-single",
		}
	}
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
	Train(train::Arguments),
	#[command(
		about = "Run inference or evaluate a trained laser intensity model"
	)]
	Infer(infer::Arguments),
}

/// The main entry-point.
///
/// # Errors
///
/// If starting the training or inference fails for any reason.
pub fn run() -> Result<(), Box<dyn Error>> {
	match Cli::parse().command {
		CliCommand::Train(arguments) => train::run(&arguments),
		CliCommand::Infer(arguments) => infer::run(&arguments),
	}
}
