//! Prepare a CLI training output directory without damaging resume data.

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

/// Validate whether a training output directory can be used.
///
/// # Errors
///
/// Returns an error when paths are invalid, overlap, or an existing output
/// would be overwritten without permission.
pub fn validate(
	output_dir: &Path,
	resume_dir: Option<&Path>,
	force: bool,
) -> Result<(), Box<dyn Error>> {
	let output_path = canonical_output_path(output_dir)?;
	if output_dir.exists() && !output_dir.is_dir() {
		return Err(IoError::new(
			ErrorKind::InvalidInput,
			"output path exists but is not a directory",
		)
		.into());
	}
	if let Some(resume_dir) = resume_dir {
		let resume_path = fs::canonicalize(resume_dir)?;
		if !resume_path.is_dir() {
			return Err(IoError::new(
				ErrorKind::InvalidInput,
				"resume path is not a directory",
			)
			.into());
		}
		if output_path == resume_path {
			return Ok(());
		}
		if output_path.starts_with(&resume_path)
			|| resume_path.starts_with(&output_path)
		{
			return Err(IoError::new(
				ErrorKind::InvalidInput,
				"output directory must not overlap the resume source",
			)
			.into());
		}
		if output_dir.exists() {
			return Err(IoError::new(
				ErrorKind::AlreadyExists,
				"resume output directory already exists; choose another output or resume in place",
			)
			.into());
		}
	}
	if resume_dir.is_none() && output_dir.exists() && !force {
		return Err(IoError::new(
			ErrorKind::AlreadyExists,
			"output directory already exists; pass --force to overwrite it",
		)
		.into());
	}
	Ok(())
}

/// Check whether two directory paths resolve to overlapping locations.
///
/// # Errors
///
/// Returns an error when either path cannot be resolved against an existing
/// filesystem ancestor.
pub fn paths_overlap(
	first: &Path,
	second: &Path,
) -> Result<bool, Box<dyn Error>> {
	let first = canonical_output_path(first)?;
	let second = canonical_output_path(second)?;
	Ok(first.starts_with(&second) || second.starts_with(&first))
}

/// Prepare a previously validated training output directory.
///
/// # Errors
///
/// Returns an error if validation or filesystem preparation fails.
pub fn prepare(
	output_dir: &Path,
	resume_dir: Option<&Path>,
	force: bool,
) -> Result<(), Box<dyn Error>> {
	validate(output_dir, resume_dir, force)?;
	if let Some(resume_dir) = resume_dir
		&& canonical_output_path(output_dir)? == fs::canonicalize(resume_dir)?
	{
		return Ok(());
	}
	if output_dir.exists() && force && resume_dir.is_none() {
		fs::remove_dir_all(output_dir)?;
	}
	fs::create_dir_all(output_dir)?;
	Ok(())
}

fn canonical_output_path(output_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
	let absolute_path = if output_dir.is_absolute() {
		output_dir.to_path_buf()
	} else {
		env::current_dir()?.join(output_dir)
	};
	let normalised_path = normalise_path(&absolute_path);
	if normalised_path.exists() {
		return Ok(fs::canonicalize(normalised_path)?);
	}
	let mut ancestor = normalised_path.as_path();
	let mut suffix = Vec::new();
	while !ancestor.exists() {
		let component = ancestor.file_name().ok_or_else(|| {
			IoError::new(ErrorKind::InvalidInput, "invalid output path")
		})?;
		suffix.push(component.to_owned());
		ancestor = ancestor.parent().ok_or_else(|| {
			IoError::new(ErrorKind::InvalidInput, "invalid output path")
		})?;
	}
	let mut canonical = fs::canonicalize(ancestor)?;
	for component in suffix.iter().rev() {
		canonical.push(component);
	}
	Ok(canonical)
}

fn normalise_path(path: &Path) -> PathBuf {
	let mut normalised = PathBuf::new();
	for component in path.components() {
		match component {
			std::path::Component::Prefix(_) | std::path::Component::RootDir => {
				normalised.push(component.as_os_str());
			},
			std::path::Component::CurDir => {},
			std::path::Component::ParentDir => {
				_ = normalised.pop();
			},
			std::path::Component::Normal(name) => normalised.push(name),
		}
	}
	normalised
}

#[cfg(test)]
mod tests {
	use std::{
		env,
		fs,
		process,
		sync::atomic::{
			AtomicUsize,
			Ordering,
		},
	};

	use super::*;

	static NEXT_TEST_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

	fn create_test_directory() -> PathBuf {
		let directory = env::temp_dir().join(format!(
			"cnn-train-cli-{}-{}",
			process::id(),
			NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed),
		));
		fs::create_dir_all(&directory)
			.expect("test directory should be created");
		directory
	}

	#[test]
	fn existing_output_requires_force_and_is_preserved() {
		let root = create_test_directory();
		let output = root.join("output");
		fs::create_dir_all(&output).expect("output should be created");
		let sentinel = output.join("keep.txt");
		fs::write(&sentinel, "preserve").expect("sentinel should be written");

		assert!(prepare(&output, None, false).is_err());
		assert!(sentinel.exists());

		fs::remove_dir_all(root).expect("test directory should be removed");
	}

	#[test]
	fn force_replaces_existing_output_directory() {
		let root = create_test_directory();
		let output = root.join("output");
		fs::create_dir_all(&output).expect("output should be created");
		let sentinel = output.join("remove.txt");
		fs::write(&sentinel, "remove").expect("sentinel should be written");

		prepare(&output, None, true)
			.expect("forced output preparation should succeed");

		assert!(!sentinel.exists());
		assert!(output.is_dir());
		fs::remove_dir_all(root).expect("test directory should be removed");
	}

	#[test]
	fn output_nested_in_resume_source_is_rejected() {
		let root = create_test_directory();
		let source = root.join("source");
		fs::create_dir_all(&source).expect("source should be created");
		let checkpoint = source.join("checkpoint");
		fs::create_dir_all(&checkpoint)
			.expect("checkpoint directory should be created");

		assert!(prepare(&source.join("output"), Some(&source), true).is_err());
		assert!(checkpoint.is_dir());

		fs::remove_dir_all(root).expect("test directory should be removed");
	}

	#[test]
	fn resume_can_target_its_source_even_with_force() {
		let root = create_test_directory();
		let checkpoint = root.join("checkpoint");
		fs::create_dir_all(&checkpoint)
			.expect("checkpoint directory should be created");
		let sentinel = checkpoint.join("keep.txt");
		fs::write(&sentinel, "preserve").expect("sentinel should be written");

		prepare(&root, Some(&root), true)
			.expect("in-place resume preparation should succeed");

		assert!(sentinel.is_file());
		fs::remove_dir_all(root).expect("test directory should be removed");
	}

	#[test]
	fn resume_does_not_force_overwrite_a_different_existing_target() {
		let root = create_test_directory();
		let source = root.join("source");
		let output = root.join("output");
		fs::create_dir_all(&source).expect("resume source should be created");
		fs::create_dir_all(&output).expect("output should be created");
		let sentinel = output.join("keep.txt");
		fs::write(&sentinel, "preserve").expect("sentinel should be written");

		assert!(prepare(&output, Some(&source), true).is_err());
		assert!(sentinel.is_file());
		fs::remove_dir_all(root).expect("test directory should be removed");
	}

	#[test]
	fn overlapping_paths_are_detected_after_canonicalization() {
		let root = create_test_directory();
		let first = root.join("artefacts");
		let nested = first.join("reports");
		let alias = root.join("missing/../artefacts");

		assert!(paths_overlap(&first, &nested).expect("paths should resolve"));
		assert!(
			paths_overlap(&alias, &nested)
				.expect("normalised paths should resolve")
		);
		fs::remove_dir_all(root).expect("test directory should be removed");
	}
}
