//! Model inference over individual samples or complete datasets.

#![expect(clippy::expect_used, reason = "No need to worry about panic here")]

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

/// Run inference for an in-memory collection of mapped samples.
///
/// # Panics
///
/// Panics if the artifact directory does not contain a readable model
/// configuration and model export, or if predictions cannot be converted to
/// the backend's floating-point element type.
pub fn infer<B>(
	artefact_dir: &str,
	device: &B::Device,
	items: Vec<LoadedLaserItem<B>>,
) -> InferenceResults<B::FloatElem>
where
	B: Backend,
{
	let model = load_model(artefact_dir, device);
	let metadata = collect_metadata(&items);
	let batch = create_batch(items, device);
	let predictions = predict(&model, batch);
	assemble_results(metadata, predictions)
}

/// Infer every sample in a dataset using bounded batches.
///
/// # Errors
///
/// Returns an error if a dataset entry is unexpectedly unavailable.
///
/// # Panics
///
/// Panics if the artifact directory does not contain a readable model
/// configuration and model export, or if predictions cannot be converted to
/// the backend's floating-point element type.
pub fn infer_dataset<B, D>(
	artefact_dir: &str,
	device: &B::Device,
	dataset: &D,
	batch_size: NonZeroUsize,
) -> Result<InferenceResults<B::FloatElem>, IoError>
where
	B: Backend,
	D: Dataset<LoadedLaserItem<B>>,
{
	let model = load_model(artefact_dir, device);
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
		let predictions = predict(&model, create_batch(items, device));
		samples.extend(assemble_results(metadata, predictions).0);
	}

	Ok(InferenceResults(samples))
}

fn load_model<B>(
	artefact_dir: &str,
	device: &B::Device,
) -> LaserModel<B>
where
	B: Backend,
{
	let config = TrainingConfig::load(format!("{artefact_dir}/config.json"))
		.expect("Config should exist for the model; run train first");
	persistence::load_model(
		&config.model,
		Path::new(artefact_dir).join("model.bpk"),
		device,
	)
	.expect("Trained model should exist; run train first")
}

fn collect_metadata<B>(
	items: &[LoadedLaserItem<B>]
) -> Vec<SampleMetadata<B::FloatElem>>
where
	B: Backend,
{
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

fn create_batch<B>(
	items: Vec<LoadedLaserItem<B>>,
	device: &B::Device,
) -> LaserBatch<B>
where
	B: Backend,
{
	LaserBatcher.batch(items, device)
}

fn predict<B>(
	model: &LaserModel<B>,
	batch: LaserBatch<B>,
) -> Vec<B::FloatElem>
where
	B: Backend,
{
	model
		.forward(batch.images)
		.flatten::<1>(0_i32, 1_i32)
		.into_data()
		.to_vec::<B::FloatElem>()
		.expect("model should return one prediction per sample")
}

fn assemble_results<T>(
	metadata: Vec<SampleMetadata<T>>,
	predictions: Vec<T>,
) -> InferenceResults<T>
where
	T: Element,
{
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
