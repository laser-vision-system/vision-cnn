//! Resolve source, output, and optional report directories for training.

use std::{
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

use crate::paths;

const DEFAULT_OUTPUT_DIR: &str = "./output";
const DEFAULT_REPORT_DIR: &str = "./reports";

#[expect(
	clippy::struct_field_names,
	reason = "Explicitly communicate that these are directories"
)]
pub struct TrainPaths {
	pub output_dir: PathBuf,
	pub source_dir: Option<PathBuf>,
	pub report_dir: Option<PathBuf>,
}

pub fn resolve(
	source: Option<&Path>,
	output_name: Option<&str>,
	evaluate: bool,
	dry_run: bool,
) -> Result<TrainPaths, Box<dyn Error>> {
	let output_base = if source.is_some() || !dry_run {
		paths::resolve_base_dir("VISION_CNN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)?
	} else {
		PathBuf::from(DEFAULT_OUTPUT_DIR)
	};
	let source_dir = source
		.map(|source| resolve_source(&output_base, source))
		.transpose()?;
	let output_dir = resolve_output(
		&output_base,
		output_name,
		source_dir.as_deref(),
		dry_run,
	)?;
	let report_dir = resolve_report(&output_dir, evaluate)?;
	Ok(TrainPaths {
		output_dir,
		source_dir,
		report_dir,
	})
}

fn resolve_output(
	output_base: &Path,
	output_name: Option<&str>,
	source_dir: Option<&Path>,
	dry_run: bool,
) -> Result<PathBuf, Box<dyn Error>> {
	if dry_run {
		return Ok(PathBuf::from("./temp").join(paths::timestamp()?));
	}
	if let Some(name) = output_name {
		paths::validate_leaf(name, "output name")?;
		return Ok(output_base.join(name));
	}
	if let Some(source_dir) = source_dir {
		return Ok(source_dir.to_path_buf());
	}
	Err(IoError::new(
		ErrorKind::InvalidInput,
		"OUTPUT_NAME is required unless --dry-run is used",
	)
	.into())
}

fn resolve_report(
	output_dir: &Path,
	evaluate: bool,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
	if !evaluate {
		return Ok(None);
	}
	let report_base =
		paths::resolve_base_dir("VISION_CNN_REPORT_DIR", DEFAULT_REPORT_DIR)?;
	let output_name = output_dir
		.file_name()
		.and_then(|name| name.to_str())
		.ok_or_else(|| {
			IoError::new(
				ErrorKind::InvalidInput,
				"output directory must have a valid name",
			)
		})?;
	Ok(Some(report_base.join(output_name)))
}

fn resolve_source(
	base_dir: &Path,
	source: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
	let path = if source.is_absolute() {
		source.to_path_buf()
	} else {
		let name = source.to_str().ok_or_else(|| {
			IoError::new(
				ErrorKind::InvalidInput,
				"model directory is not UTF-8",
			)
		})?;
		paths::validate_leaf(name, "model directory")?;
		base_dir.join(name)
	};
	let canonical = fs::canonicalize(path)?;
	if !canonical.is_dir() {
		return Err(IoError::new(
			ErrorKind::InvalidInput,
			"model path is not a directory",
		)
		.into());
	}
	Ok(canonical)
}
