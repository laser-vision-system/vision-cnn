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

pub(super) fn prepare(
	output_dir: &Path,
	resume_dir: Option<&Path>,
	force: bool,
) -> Result<(), Box<dyn Error>> {
	create_parent(output_dir, resume_dir)?;
	validate_paths(output_dir, resume_dir)?;
	remove_existing(output_dir, force)?;
	fs::create_dir_all(output_dir)?;
	Ok(())
}

fn validate_paths(
	output_dir: &Path,
	resume_dir: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
	let Some(resume_dir) = resume_dir else {
		return Ok(());
	};
	let resume_path = fs::canonicalize(resume_dir)?;
	let output_path = canonical_output_path(output_dir)?;
	if output_path.starts_with(&resume_path)
		|| resume_path.starts_with(&output_path)
	{
		return Err(IoError::new(
			ErrorKind::InvalidInput,
			"output directory must not be the resume source or overlap it",
		)
		.into());
	}
	Ok(())
}

fn canonical_output_path(output_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
	if output_dir.exists() {
		return Ok(fs::canonicalize(output_dir)?);
	}
	let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
	let parent = if parent.as_os_str().is_empty() {
		Path::new(".")
	} else {
		parent
	};
	let filename = output_dir.file_name().ok_or_else(|| {
		IoError::new(ErrorKind::InvalidInput, "invalid output path")
	})?;
	Ok(fs::canonicalize(parent)?.join(filename))
}

fn create_parent(
	output_dir: &Path,
	resume_dir: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
	if resume_dir.is_some()
		&& let Some(parent) = output_dir
			.parent()
			.filter(|parent| !parent.as_os_str().is_empty())
	{
		fs::create_dir_all(parent)?;
	}
	Ok(())
}

fn remove_existing(
	output_dir: &Path,
	force: bool,
) -> Result<(), Box<dyn Error>> {
	if !output_dir.exists() {
		return Ok(());
	}
	if !force {
		return Err(IoError::new(
			ErrorKind::AlreadyExists,
			"output directory already exists; pass --force to overwrite it",
		)
		.into());
	}
	fs::remove_dir_all(output_dir)?;
	Ok(())
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
}
