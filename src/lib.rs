//! Command-line interface for the vision-cnn library.

#![cfg_attr(
	test,
	allow(
		clippy::print_stdout,
		clippy::print_stderr,
		clippy::use_debug,
		clippy::panic,
		clippy::unwrap_used,
		clippy::unwrap_in_result,
		clippy::expect_used,
		clippy::indexing_slicing,
		reason = "Tests are allowed to perform all of these operations"
	)
)]

mod infer;
mod output;
mod paths;
mod train;

use std::error::Error;

use clap::{
	Parser,
	Subcommand,
	ValueEnum,
};
use vision_cnn::data::DatasetSource;

#[derive(Clone, Copy, Debug, ValueEnum)]
/// A dataset source selectable by the command-line interface.
pub enum DatasetChoice {
	/// The first synthetic dataset.
	Synthetic1,
	/// The second synthetic dataset.
	Synthetic2,
	/// The analogous laser dataset.
	Analogous,
	/// The in-lab single-laser dataset.
	InLabSingle,
}

impl DatasetChoice {
	/// Convert this command-line value to the internal dataset source.
	pub const fn source(self) -> DatasetSource {
		match self {
			Self::Synthetic1 => DatasetSource::Synthetic1,
			Self::Synthetic2 => DatasetSource::Synthetic2,
			Self::Analogous => DatasetSource::Analogous,
			Self::InLabSingle => DatasetSource::InLabSingle,
		}
	}

	/// Return the command-line name for this dataset.
	pub const fn name(self) -> &'static str {
		match self {
			Self::Synthetic1 => "synthetic1",
			Self::Synthetic2 => "synthetic2",
			Self::Analogous => "analogous",
			Self::InLabSingle => "in-lab-single",
		}
	}
}

#[derive(Debug, Parser)]
/// Parsed arguments for the command-line application.
#[command(
	name = "cnn",
	about = "Train and evaluate laser intensity models",
	subcommand_required = true,
	arg_required_else_help = true
)]
struct VisionCnnCli {
	#[command(subcommand)]
	command: CliCommand,
}

#[derive(Debug, Subcommand)]
/// A supported operation exposed by the command-line application.
enum CliCommand {
	#[command(about = "Train a laser intensity model")]
	Train(train::TrainArgs),
	#[command(
		about = "Run inference or evaluate a trained laser intensity model"
	)]
	Infer(infer::InferArgs),
}

/// Start the selected command workflow.
///
/// # Errors
///
/// Returns an error when preparing or executing the selected workflow fails.
pub fn run() -> Result<(), Box<dyn Error>> {
	match VisionCnnCli::parse().command {
		CliCommand::Train(args) => train::run(args),
		CliCommand::Infer(args) => infer::run(&args),
	}
}

#[cfg(test)]
mod tests {
	use clap::Parser as _;

	use super::{
		CliCommand,
		VisionCnnCli,
	};

	#[test]
	fn parses_both_command_subcommands() {
		let train =
			VisionCnnCli::try_parse_from(["cnn", "train", "new", "artefacts"])
				.expect("train arguments should parse");
		assert!(matches!(train.command, CliCommand::Train(_)));

		let infer = VisionCnnCli::try_parse_from(["cnn", "infer", "artefacts"])
			.expect("inference arguments should parse");
		assert!(matches!(infer.command, CliCommand::Infer(_)));
	}

	#[test]
	fn train_requires_a_mode_and_new_requires_a_name_unless_dry_run() {
		VisionCnnCli::try_parse_from(["cnn", "train"])
			.expect_err("train mode should be required");
		VisionCnnCli::try_parse_from(["cnn", "train", "new"])
			.expect_err("new model name should be required");
		VisionCnnCli::try_parse_from(["cnn", "train", "new", "--dry-run"])
			.expect("dry run should not require a model name");
		VisionCnnCli::try_parse_from(["cnn", "train", "resume", "model_1"])
			.expect("resume should accept a source model");
	}

	#[test]
	fn train_accepts_resume_target_and_in_memory_evaluation_flags() {
		let parsed = VisionCnnCli::try_parse_from([
			"cnn",
			"train",
			"resume",
			"model_1",
			"-o",
			"model_2",
			"--dry-run",
			"--evaluate",
			"--seed",
			"18",
		])
		.expect("training options should parse");
		assert!(matches!(parsed.command, CliCommand::Train(_)));
	}

	#[test]
	fn new_and_resume_are_separate_train_modes() {
		VisionCnnCli::try_parse_from(["cnn", "train", "new", "model_1"])
			.unwrap();
		VisionCnnCli::try_parse_from(["cnn", "train", "resume", "model_1"])
			.unwrap();
	}
}
