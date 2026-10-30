//! Burnpack model export and loading.

use std::path::Path;

use burn::{
	module::Module as _,
	store::RecordError,
	tensor::Device,
};

use crate::model::{
	LaserModel,
	LaserModelConfig,
};

/// Export model tensors in Burn's native Burnpack format.
///
/// # Errors
///
/// If the model is unable to be saved into the store path.
pub fn export_model(
	model: LaserModel,
	path: impl AsRef<Path>,
) -> Result<(), RecordError> {
	model.save_file(path)
}

/// Load a Burnpack model export using the provided model configuration.
///
/// # Errors
///
/// If the model is unable to be loaded from the store path.
pub fn load_model(
	config: &LaserModelConfig,
	path: impl AsRef<Path>,
	device: &Device,
) -> Result<LaserModel, RecordError> {
	let model = config.init(device);
	model.try_load_file(path)
}

#[cfg(test)]
mod tests {
	use std::{
		env,
		fs,
		path::PathBuf,
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
			"cnn-burnpack-{}-{}",
			process::id(),
			NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed),
		));
		fs::create_dir_all(&directory)
			.expect("test directory should be created");
		directory
	}

	#[test]
	fn exported_model_can_be_loaded() {
		let directory = create_test_directory();
		let path = directory.join("model.bpk");
		let device = Device::default();
		let config = LaserModelConfig::new();
		let model = config.init(&device);

		export_model(model, &path).expect("model export should succeed");
		let loaded = load_model(&config, &path, &device);

		assert!(loaded.is_ok(), "exported model should load successfully");
		assert!(path.is_file(), "Burnpack export should be written");
		fs::remove_dir_all(directory)
			.expect("test directory should be removed");
	}
}
