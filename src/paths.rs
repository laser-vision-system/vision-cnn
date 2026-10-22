//! Shared path validation and environment-backed directory resolution.

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

pub fn resolve_base_dir(
	env_var: &str,
	default: &str,
) -> Result<PathBuf, Box<dyn Error>> {
	let path =
		PathBuf::from(env::var(env_var).unwrap_or_else(|_| default.to_owned()));
	if !path.exists() {
		return Ok(path);
	}

	if !path.is_dir() {
		return Err(IoError::new(
			ErrorKind::InvalidInput,
			format!("{env_var} must name a directory"),
		)
		.into());
	}
	Ok(fs::canonicalize(path)?)
}

// TODO: Use dedicated error type to remove the need for the label parameter.
pub fn validate_leaf(
	value: &str,
	label: &str,
) -> Result<(), Box<dyn Error>> {
	let path = Path::new(value);
	if value.is_empty()
		|| path.components().count() != 1
		|| path.file_name().is_none_or(|name| name != value)
	{
		return Err(IoError::new(
			ErrorKind::InvalidInput,
			format!("{label} must be a single non-empty path component"),
		)
		.into());
	}
	Ok(())
}

pub fn timestamp() -> Result<String, time::error::Format> {
	time::OffsetDateTime::now_local()
		.unwrap_or_else(|_| time::OffsetDateTime::now_utc())
		.format(&time::macros::format_description!(
			"[year]-[month]-[day]_[hour]:[minute]:[second][offset_hour sign:mandatory]"
		))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn accepts_only_non_empty_leaf_names() {
		assert!(matches!(validate_leaf("model_1", "name"), Ok(())));
		assert!(validate_leaf("", "name").is_err());
		assert!(validate_leaf("../model_1", "name").is_err());
		assert!(validate_leaf("folder/model_1", "name").is_err());
	}

	#[test]
	fn timestamp_uses_the_correct_format() {
		let value =
			timestamp().expect("format should be validated by the time macro");
		assert_eq!(value.len(), 22);
		let mut chars = value.chars();
		assert_eq!(chars.nth(4), Some('-'));
		assert_eq!(chars.nth(2), Some('-'));
		assert_eq!(chars.nth(2), Some('_'));
		assert_eq!(chars.nth(2), Some(':'));
		assert_eq!(chars.nth(2), Some(':'));
		assert_eq!(chars.nth(2), Some('+'));
		assert!(value.char_indices().all(|(index, byte)| {
			byte.is_ascii_digit() ^ [4, 7, 10, 13, 16, 19].contains(&index)
		}));
	}
}
