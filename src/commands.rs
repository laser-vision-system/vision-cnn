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
/// A dataset source selectable by the command-line interface.
pub enum DatasetChoice {
	#[value(name = "synthetic1")]
	Synthetic1,
	#[value(name = "synthetic2")]
	Synthetic2,
	Analogous,
	#[value(name = "in-lab-single")]
	InLabSingle,
}

impl DatasetChoice {
	/// Convert this command-line value to the internal dataset source.
	pub const fn source(self) -> vision_cnn::data::DatasetSource {
		match self {
			Self::Synthetic1 => vision_cnn::data::DatasetSource::Synthetic1,
			Self::Synthetic2 => vision_cnn::data::DatasetSource::Synthetic2,
			Self::Analogous => vision_cnn::data::DatasetSource::Analogous,
			Self::InLabSingle => vision_cnn::data::DatasetSource::InLabSingle,
		}
	}

	/// Return the stable command-line name for this dataset.
	pub const fn name(self) -> &'static str {
		match self {
			Self::Synthetic1 => "synthetic1",
			Self::Synthetic2 => "synthetic2",
			Self::Analogous => "analogous",
			Self::InLabSingle => "in-lab-single",
		}
	}
}

#[derive(Parser)]
/// Parsed arguments for the command-line application.
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
/// A supported operation exposed by the command-line application.
enum CliCommand {
	#[command(about = "Train a laser intensity model")]
	Train(train::Arguments),
	#[command(
		about = "Run inference or evaluate a trained laser intensity model"
	)]
	Infer(infer::Arguments),
}

/// Start the selected command workflow.
///
/// # Errors
///
/// Returns an error when preparing or executing the selected workflow fails.
pub fn run() -> Result<(), Box<dyn Error>> {
	match Cli::parse().command {
		CliCommand::Train(arguments) => train::run(&arguments),
		CliCommand::Infer(arguments) => infer::run(&arguments),
	}
}

#[cfg(test)]
mod tests {
	use clap::Parser as _;

	use super::{
		Cli,
		CliCommand,
	};

	#[test]
	fn parses_both_command_subcommands() {
		let train = Cli::try_parse_from(["cnn", "train", "artifacts"])
			.expect("train arguments should parse");
		assert!(matches!(train.command, CliCommand::Train(_)));

		let infer = Cli::try_parse_from(["cnn", "infer", "artifacts"])
			.expect("inference arguments should parse");
		assert!(matches!(infer.command, CliCommand::Infer(_)));
	}
}
