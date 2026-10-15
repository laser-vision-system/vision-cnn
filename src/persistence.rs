//! Burnpack model export and loading.

use std::path::Path;

use burn::prelude::Backend;
use burn_store::{BurnpackError, BurnpackStore, ModuleSnapshot as _};

use crate::model::{LaserModel, LaserModelConfig};

/// Export model tensors in Burn's native Burnpack format.
pub fn export_model<B, P>(
	model: &LaserModel<B>,
	path: P,
) -> Result<(), BurnpackError>
where
	B: Backend,
	P: AsRef<Path>,
{
	let mut store = BurnpackStore::from_file(path).overwrite(true);
	model.save_into(&mut store)
}

/// Load a Burnpack model export using the provided model configuration.
pub fn load_model<B, P>(
	config: &LaserModelConfig,
	path: P,
	device: &B::Device,
) -> Result<LaserModel<B>, BurnpackError>
where
	B: Backend,
	P: AsRef<Path>,
{
	let mut model = config.init::<B>(device);
	let mut store = BurnpackStore::from_file(path);
	let _apply_result = model.load_from(&mut store)?;
	Ok(model)
}

#[cfg(test)]
mod tests {
	use std::{
		env, fs,
		path::PathBuf,
		process,
		sync::atomic::{AtomicUsize, Ordering},
	};

	use burn::backend::Flex;

	use super::*;

	type Device = burn::prelude::Device<Flex>;

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
		let model = config.init::<Flex<f32>>(&device);

		export_model(&model, &path).expect("model export should succeed");
		let loaded = load_model::<Flex<f32>, _>(&config, &path, &device);

		assert!(loaded.is_ok(), "exported model should load successfully");
		assert!(path.is_file(), "Burnpack export should be written");
		fs::remove_dir_all(directory)
			.expect("test directory should be removed");
	}
}
