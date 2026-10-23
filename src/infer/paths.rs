//! Resolve and validate model and report paths for inference.

use std::{
	env,
	error::Error,
	fs,
	io::{
		Error as IoError,
		ErrorKind,
	},
	path::{
		Path,
		PathBuf,
	},
};

use super::args::InferArgs;
use crate::{
	output,
	paths,
};

pub struct InferPaths {
	pub model_dir: PathBuf,
	pub model_name: String,
	pub report_dir: Option<PathBuf>,
}

pub fn resolve(arguments: &InferArgs) -> Result<InferPaths, Box<dyn Error>> {
	let output_base =
		paths::resolve_base_dir("VISION_CNN_OUTPUT_DIR", "./output")?;
	let model_dir = resolve_model_dir(&output_base, &arguments.model_dir)?;
	let model_name = arguments
		.model_name
		.clone()
		.or_else(|| env::var("VISION_CNN_MODEL_NAME").ok())
		// TODO: Better way to handle shared fallback between train and infer.
		.unwrap_or_else(|| "model".to_owned());
	paths::validate_leaf(&model_name, "model name")?;
	let report_dir = if arguments.evaluate_all {
		let report_base =
			paths::resolve_base_dir("VISION_CNN_REPORT_DIR", "./reports")?;
		let model_directory_name = model_dir.file_name().ok_or_else(|| {
			IoError::new(
				ErrorKind::InvalidInput,
				"model directory must have a name for the default report path",
			)
		})?;
		Some(
			arguments
				.report_dir
				.clone()
				.unwrap_or_else(|| report_base.join(model_directory_name)),
		)
	} else {
		None
	};
	if let Some(report_dir) = &report_dir {
		output::validate(report_dir, None, arguments.force)?;
		if output::paths_overlap(report_dir, &model_dir)? {
			return Err(IoError::new(
				ErrorKind::InvalidInput,
				"report directory must not overlap the model directory",
			)
			.into());
		}
	}
	Ok(InferPaths {
		model_dir,
		model_name,
		report_dir,
	})
}

pub fn prepare_reports(
	paths: &InferPaths,
	force: bool,
) -> Result<(), Box<dyn Error>> {
	if let Some(report_dir) = &paths.report_dir {
		output::prepare(report_dir, None, force)?;
	}
	Ok(())
}

fn resolve_model_dir(
	base_dir: &Path,
	model_dir: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
	let name = model_dir.to_str().ok_or_else(|| {
		IoError::new(ErrorKind::InvalidInput, "model directory is not UTF-8")
	})?;
	paths::validate_leaf(name, "model directory")?;
	let canonical = fs::canonicalize(base_dir.join(name))?;
	if !canonical.is_dir() {
		return Err(IoError::new(
			ErrorKind::InvalidInput,
			"model path is not a directory",
		)
		.into());
	}
	Ok(canonical)
}
