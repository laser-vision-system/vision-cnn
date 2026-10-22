//! Model inference over individual samples or complete datasets.

#![expect(clippy::expect_used, reason = "No need to worry about panic here")]
#![expect(
	clippy::inline_trait_bounds,
	reason = "Only used with a single generic parameter"
)]

pub mod result;

use std::{
	io::{
		Error as IoError,
		ErrorKind,
	},
	num::NonZeroUsize,
	path::Path,
};

use burn::{
	data::{
		dataloader::batcher::Batcher as _,
		dataset::Dataset,
	},
	prelude::*,
	tensor::Element,
};
use result::{
	InferenceResults,
	SampleInference,
};

use crate::{
	data::{
		LaserBatch,
		LaserBatcher,
		LoadedLaserItem,
	},
	model::LaserModel,
	persistence,
	training::TrainingConfig,
};

struct SampleMetadata<T> {
	image_path: String,
	expected_intensity: T,
	position: Option<String>,
	source_fields: Vec<(&'static str, String)>,
}

/// Run inference for an in-memory collection of mapped samples using the passed
/// model.
///
/// # Panics
///
/// Panics if predictions cannot be converted to the backend's floating-point
/// element type.
pub fn infer<B: Backend>(
	model: &LaserModel<B>,
	device: &B::Device,
	items: Vec<LoadedLaserItem<B>>,
) -> InferenceResults<B::FloatElem> {
	let metadata = collect_metadata(&items);
	let batch = create_batch(items, device);
	let predictions = predict(model, batch);
	assemble_results(metadata, predictions)
}

/// Infer every sample in a dataset using the passed model and
/// batch size.
///
/// # Errors
///
/// Returns an error if a dataset entry is unexpectedly unavailable.
///
/// # Panics
///
/// Panics if predictions cannot be converted to the backend's floating-point
/// element type.
pub fn infer_dataset<B: Backend>(
	model: &LaserModel<B>,
	device: &B::Device,
	dataset: &impl Dataset<LoadedLaserItem<B>>,
	batch_size: NonZeroUsize,
) -> Result<InferenceResults<B::FloatElem>, IoError> {
	let dataset_len = dataset.len();
	let mut samples = Vec::with_capacity(dataset_len);

	for start in (0..dataset_len).step_by(batch_size.get()) {
		let end = start.saturating_add(batch_size.get()).min(dataset_len);
		let items = (start..end)
			.map(|index| {
				dataset.get(index).ok_or_else(|| {
					IoError::new(
						ErrorKind::InvalidData,
						format!("dataset sample {index} is unavailable"),
					)
				})
			})
			.collect::<Result<Vec<_>, _>>()?;
		let metadata = collect_metadata(&items);
		let predictions = predict(model, create_batch(items, device));
		samples.extend(assemble_results(metadata, predictions).0);
	}

	Ok(InferenceResults(samples))
}

/// Load a named exported model and its training configuration from an artefact
/// directory.
///
/// # Panics
///
/// Panics if the configuration or named model export cannot be loaded.
pub fn load_model<B: Backend>(
	artefact_dir: &str,
	model_name: &str,
	device: &B::Device,
) -> LaserModel<B> {
	let config = TrainingConfig::load(format!("{artefact_dir}/config.json"))
		.expect("Config should exist for the model; run train first");
	persistence::load_model(
		&config.model,
		Path::new(artefact_dir).join(model_name),
		device,
	)
	.expect("Trained model should exist; run train first")
}

fn collect_metadata<B: Backend>(
	items: &[LoadedLaserItem<B>]
) -> Vec<SampleMetadata<B::FloatElem>> {
	items
		.iter()
		.map(|item| {
			SampleMetadata {
				image_path: item.image_path.clone(),
				expected_intensity: item.intensity,
				position: item.position.clone(),
				source_fields: item.source_fields.clone(),
			}
		})
		.collect()
}

fn create_batch<B: Backend>(
	items: Vec<LoadedLaserItem<B>>,
	device: &B::Device,
) -> LaserBatch<B> {
	LaserBatcher.batch(items, device)
}

fn predict<B: Backend>(
	model: &LaserModel<B>,
	batch: LaserBatch<B>,
) -> Vec<B::FloatElem> {
	model
		.forward(batch.images)
		.flatten::<1>(0_i32, 1_i32)
		.into_data()
		.to_vec::<B::FloatElem>()
		.expect("model should return one prediction per sample")
}

fn assemble_results<T: Element>(
	metadata: Vec<SampleMetadata<T>>,
	predictions: Vec<T>,
) -> InferenceResults<T> {
	let samples = metadata
		.into_iter()
		.zip(predictions)
		.map(|(sample, predicted_intensity)| {
			SampleInference {
				image_path: sample.image_path,
				position: sample.position,
				expected_intensity: sample.expected_intensity,
				predicted_intensity,
				source_fields: sample.source_fields,
			}
		})
		.collect();
	InferenceResults(samples)
}

#[cfg(test)]
mod tests {
	use burn_flex::Flex;

	use super::*;
	use crate::data::ImageType;

	#[test]
	fn evaluation_metadata_preserves_mapped_float_targets() {
		let items = [LoadedLaserItem::<Flex> {
			image: ImageType::new(1, 1),
			image_path: "dummy/sample.png".to_owned(),
			intensity: 7.0,
			position: Some("sample-a".to_owned()),
			source_fields: vec![("source", "dummy".to_owned())],
		}];

		let metadata = collect_metadata(&items);

		assert_eq!(metadata.len(), 1);
		assert!(metadata.first().is_some_and(|sample| {
			(sample.expected_intensity - 7_f32).abs() < f32::EPSILON
				&& sample.position.as_deref() == Some("sample-a")
				&& sample.source_fields == [("source", "dummy".to_owned())]
		}));
	}
}
