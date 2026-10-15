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
use dataset_loader::item::NumericLike;
use result::{
	InferenceResults,
	SampleInference,
};

use crate::{
	data::{
		IntensityValType,
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

pub fn infer<B, T>(
	artefact_dir: &str,
	device: &B::Device,
	items: Vec<LoadedLaserItem<T>>,
) -> InferenceResults<T>
where
	B: Backend,
	T: NumericLike + Into<IntensityValType> + Element,
{
	let model = load_model::<B>(artefact_dir, device);
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
pub fn infer_dataset<B, T, D>(
	artefact_dir: &str,
	device: &B::Device,
	dataset: &D,
	batch_size: NonZeroUsize,
) -> Result<InferenceResults<IntensityValType>, IoError>
where
	B: Backend,
	T: NumericLike + Into<IntensityValType> + Element,
	D: Dataset<LoadedLaserItem<T>>,
{
	let model = load_model::<B>(artefact_dir, device);
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
		let metadata = collect_evaluation_metadata(&items);
		let predictions =
			predict::<B, IntensityValType>(&model, create_batch(items, device));
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

fn collect_metadata<T>(items: &[LoadedLaserItem<T>]) -> Vec<SampleMetadata<T>>
where
	T: NumericLike,
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

// NOTE: This function is basically the same as the above, but less generic.
//   Can they be combined?
fn collect_evaluation_metadata<T>(
	items: &[LoadedLaserItem<T>]
) -> Vec<SampleMetadata<IntensityValType>>
where
	T: NumericLike + Into<IntensityValType>,
{
	items
		.iter()
		.map(|item| {
			SampleMetadata {
				image_path: item.image_path.clone(),
				expected_intensity: item.intensity.into(),
				position: item.position.clone(),
				source_fields: item.source_fields.clone(),
			}
		})
		.collect()
}

fn create_batch<B, T>(
	items: Vec<LoadedLaserItem<T>>,
	device: &B::Device,
) -> LaserBatch<B>
where
	B: Backend,
	T: NumericLike + Into<IntensityValType>,
{
	LaserBatcher.batch(items, device)
}

fn predict<B, T>(
	model: &LaserModel<B>,
	batch: LaserBatch<B>,
) -> Vec<T>
where
	B: Backend,
	T: NumericLike + Into<IntensityValType> + Element,
{
	model
		.forward(batch.images)
		.flatten::<1>(0_i32, 1_i32)
		.into_data()
		.to_vec::<T>()
		.expect("model should return one prediction per sample")
}

fn assemble_results<T>(
	metadata: Vec<SampleMetadata<T>>,
	predictions: Vec<T>,
) -> InferenceResults<T>
where
	T: NumericLike,
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
	use image::RgbImage;

	use super::*;

	#[test]
	fn evaluation_metadata_converts_integer_targets_without_rounding() {
		let items = [LoadedLaserItem {
			image: RgbImage::new(1, 1),
			image_path: "dummy/sample.png".to_owned(),
			intensity: 7_u8,
			position: Some("sample-a".to_owned()),
			source_fields: vec![("source", "dummy".to_owned())],
		}];

		let metadata = collect_evaluation_metadata(&items);

		assert_eq!(metadata.len(), 1);
		assert!(metadata.first().is_some_and(|sample| {
			(sample.expected_intensity - 7_f32).abs() < f32::EPSILON
				&& sample.position.as_deref() == Some("sample-a")
				&& sample.source_fields == [("source", "dummy".to_owned())]
		}));
	}
}
