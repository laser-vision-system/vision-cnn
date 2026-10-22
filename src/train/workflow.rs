//! Compose path resolution, validation, training, and model export.

use std::{
	env,
	error::Error,
	fs,
	io::{
		Error as IoError,
		ErrorKind,
	},
};

use vision_cnn::{
	persistence,
	plan,
};

use super::{
	args::{
		TrainCommonArgs,
		TrainOverrideArgs,
	},
	engine,
	paths::{
		self,
		TrainPaths,
	},
};
use crate::{
	output,
	paths::{
		timestamp,
		validate_leaf,
	},
};

const DEFAULT_MODEL_NAME: &str = "model";
const MODEL_EXTENSION: &str = ".bpk";

pub struct TrainRequest {
	pub source: Option<std::path::PathBuf>,
	pub output_name: Option<String>,
	pub options: TrainCommonArgs,
	pub overrides: TrainOverrideArgs,
}

pub fn run(request: &TrainRequest) -> Result<(), Box<dyn Error>> {
	let paths = paths::resolve(
		request.source.as_deref(),
		request.output_name.as_deref(),
		request.options.evaluate,
		request.options.dry_run,
	)?;
	let model_name = resolve_model_name(&request.options)?;
	validate_paths(&paths, &request.options)?;
	let training_plan = plan::load(
		paths.source_dir.as_deref(),
		&request.overrides.clone().into(),
	)?;
	prepare_directories(&paths, &request.options)?;
	backup_in_place_model(&paths, &model_name, request.options.dry_run)?;
	let model = engine::train(
		&paths.output_dir,
		&training_plan,
		request.options.dataset.source(),
	)?;
	if let Some(report_dir) = &paths.report_dir {
		engine::evaluate(&model, report_dir, request.options.dataset)?;
	}
	export_model(&paths, &model_name, &model, request.options.dry_run)?;
	Ok(())
}

fn resolve_model_name(
	options: &TrainCommonArgs
) -> Result<String, Box<dyn Error>> {
	if options.dry_run {
		return Ok(DEFAULT_MODEL_NAME.to_owned());
	}
	let name = options
		.model_name
		.clone()
		.or_else(|| env::var("VISION_CNN_MODEL_NAME").ok())
		.unwrap_or_else(|| DEFAULT_MODEL_NAME.to_owned());
	validate_leaf(&name, "model name")?;
	Ok(name)
}

fn validate_paths(
	paths: &TrainPaths,
	options: &TrainCommonArgs,
) -> Result<(), Box<dyn Error>> {
	let force = options.force && paths.source_dir.is_none() && !options.dry_run;
	output::validate(&paths.output_dir, paths.source_dir.as_deref(), force)?;
	if let Some(report_dir) = &paths.report_dir {
		output::validate(report_dir, None, force)?;
		let overlaps_output =
			output::paths_overlap(report_dir, &paths.output_dir)?;
		let overlaps_source = if let Some(source) = &paths.source_dir {
			output::paths_overlap(report_dir, source)?
		} else {
			false
		};
		if overlaps_output || overlaps_source {
			return Err(IoError::new(
				ErrorKind::InvalidInput,
				"report directory must not overlap training artefacts or the resume source",
			)
			.into());
		}
	}
	Ok(())
}

fn prepare_directories(
	paths: &TrainPaths,
	options: &TrainCommonArgs,
) -> Result<(), Box<dyn Error>> {
	let force = options.force && paths.source_dir.is_none() && !options.dry_run;
	output::prepare(&paths.output_dir, paths.source_dir.as_deref(), force)?;
	if let Some(report_dir) = &paths.report_dir {
		output::prepare(report_dir, None, force)?;
	}
	Ok(())
}

fn backup_in_place_model(
	paths: &TrainPaths,
	model_name: &str,
	dry_run: bool,
) -> Result<(), Box<dyn Error>> {
	if !dry_run
		&& let Some(source) = &paths.source_dir
		&& fs::canonicalize(source)? == fs::canonicalize(&paths.output_dir)?
	{
		let model = paths
			.output_dir
			.join(format!("{model_name}{MODEL_EXTENSION}"));
		if model.is_file() {
			let backup = paths.output_dir.join(format!(
				"{model_name}_{}{MODEL_EXTENSION}",
				timestamp()?
			));
			if backup.exists() {
				return Err(IoError::new(
					ErrorKind::AlreadyExists,
					"timestamped model backup already exists",
				)
				.into());
			}
			fs::rename(model, backup)?;
		}
	}
	Ok(())
}

fn export_model(
	paths: &TrainPaths,
	model_name: &str,
	model: &vision_cnn::model::LaserModel<burn::backend::Cuda>,
	dry_run: bool,
) -> Result<(), Box<dyn Error>> {
	if !dry_run {
		persistence::export_model(
			model,
			paths
				.output_dir
				.join(format!("{model_name}{MODEL_EXTENSION}")),
		)?;
	}
	Ok(())
}
