//! Regression metrics for model evaluation.

use std::{
	collections::BTreeMap,
	error::Error,
	fmt,
	io::Write,
};

use burn::tensor::Element;
use serde::Serialize;

use crate::inference::result::SampleInference;

/// A failure encountered while computing regression metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegressionMetricsError {
	/// Metrics cannot be computed without any samples.
	EmptyInput,
	/// A target or prediction was not finite.
	NonFiniteValue { sample_index: usize },
}

impl fmt::Display for RegressionMetricsError {
	fn fmt(
		&self,
		formatter: &mut fmt::Formatter<'_>,
	) -> fmt::Result {
		match self {
			Self::EmptyInput => {
				formatter.write_str("no evaluation samples provided")
			},
			Self::NonFiniteValue {
				sample_index,
			} => {
				write!(
					formatter,
					"sample {sample_index} contains a non-finite target or prediction"
				)
			},
		}
	}
}

impl Error for RegressionMetricsError {
}

/// Aggregate regression metrics computed over target/prediction pairs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RegressionMetrics {
	/// Number of samples included in the summary.
	pub sample_count: usize,
	/// Mean squared error.
	pub mean_squared_error: f64,
	/// Root mean squared error.
	pub root_mean_squared_error: f64,
	/// Mean absolute error.
	pub mean_absolute_error: f64,
	/// Mean signed error, defined as prediction minus target.
	pub mean_error: f64,
	/// Median absolute error.
	pub median_absolute_error: f64,
	/// 95th percentile absolute error using linear interpolation.
	pub percentile_95_absolute_error: f64,
	/// Coefficient of determination, or `None` when targets have zero
	/// variance.
	pub r_squared: Option<f64>,
}

/// One scored sample included in an evaluation report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvaluationSample {
	/// Zero-based position in the selected, deterministically ordered split.
	pub sample_index: usize,
	/// Canonical image path identifying this sample on the current machine.
	pub image_path: String,
	/// Optional dataset position or capture label.
	pub position: Option<String>,
	/// Expected target value.
	pub expected: f32,
	/// Model prediction.
	pub predicted: f32,
	/// Prediction minus expected value.
	pub residual: f32,
	/// Absolute residual.
	pub absolute_error: f32,
	/// Source-specific metadata fields.
	pub source_fields: BTreeMap<String, String>,
}

/// Machine-readable summary and per-sample results for one evaluation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvaluationReport {
	/// Dataset choice supplied to the inference command.
	pub dataset: String,
	/// Split used to produce the metrics.
	pub split: String,
	/// Aggregate regression metrics.
	pub metrics: RegressionMetrics,
	/// Per-sample predictions and metadata.
	pub samples: Vec<EvaluationSample>,
}

impl EvaluationReport {
	/// Create a report from ordered predictions and aggregate metrics.
	pub fn new<D, S, T>(
		dataset: D,
		split: S,
		metrics: RegressionMetrics,
		results: &[SampleInference<T>],
	) -> Self
	where
		D: Into<String>,
		S: Into<String>,
		T: Element + Into<f32>,
	{
		let samples = results
			.iter()
			.enumerate()
			.map(|(sample_index, result)| {
				let expected = result.expected_intensity.into();
				let predicted = result.predicted_intensity.into();
				let residual = predicted - expected;
				EvaluationSample {
					sample_index,
					image_path: result.image_path.clone(),
					position: result.position.clone(),
					expected,
					predicted,
					residual,
					absolute_error: residual.abs(),
					source_fields: result
						.source_fields
						.iter()
						.map(|(name, value)| {
							((*name).to_owned(), value.clone())
						})
						.collect(),
				}
			})
			.collect();

		Self {
			dataset: dataset.into(),
			split: split.into(),
			metrics,
			samples,
		}
	}

	/// Write the report summary as formatted JSON.
	///
	/// # Errors
	///
	/// Returns an error if serialization or writing fails.
	pub fn write_json<W>(
		&self,
		mut writer: W,
	) -> Result<(), Box<dyn Error>>
	where
		W: Write,
	{
		serde_json::to_writer_pretty(&mut writer, self)?;
		writer.flush()?;
		Ok(())
	}

	/// Write per-sample results as CSV, with source fields encoded as JSON.
	///
	/// # Errors
	///
	/// Returns an error if serialization or writing fails.
	pub fn write_csv<W>(
		&self,
		writer: W,
	) -> Result<(), Box<dyn Error>>
	where
		W: Write,
	{
		let mut csv_writer = csv::Writer::from_writer(writer);
		csv_writer.write_record([
			"sample_index",
			"sample_id",
			"position",
			"expected",
			"predicted",
			"residual",
			"absolute_error",
			"source_fields_json",
		])?;
		for sample in &self.samples {
			let source_fields = serde_json::to_string(&sample.source_fields)?;
			csv_writer.write_record([
				sample.sample_index.to_string(),
				sample.image_path.clone(),
				sample.position.clone().unwrap_or_default(),
				sample.expected.to_string(),
				sample.predicted.to_string(),
				sample.residual.to_string(),
				sample.absolute_error.to_string(),
				source_fields,
			])?;
		}
		csv_writer.flush()?;
		Ok(())
	}
}

/// Compute aggregate regression metrics from `(target, prediction)` pairs.
///
/// The 95th percentile uses linear interpolation between adjacent sorted
/// absolute errors. $R^2$ is undefined when every target has the same value.
///
/// # Errors
///
/// Returns [`RegressionMetricsError::EmptyInput`] when there are no pairs and
/// [`RegressionMetricsError::NonFiniteValue`] when a pair contains NaN or an
/// infinite value.
#[expect(
	clippy::cast_precision_loss,
	reason = "The sample count is bounded by the allocated in-memory evaluation pairs"
)]
pub fn regression_metrics<I>(
	samples: I
) -> Result<RegressionMetrics, RegressionMetricsError>
where
	I: IntoIterator<Item = (f64, f64)>,
{
	let mut absolute_errors = Vec::new();
	let mut squared_error_sum = 0.0;
	let mut signed_error_sum = 0.0;
	let mut target_sum = 0.0;
	let mut pairs: Vec<(f64, f64)> = Vec::new();

	for (sample_index, (target, prediction)) in samples.into_iter().enumerate()
	{
		if !target.is_finite() || !prediction.is_finite() {
			return Err(RegressionMetricsError::NonFiniteValue {
				sample_index,
			});
		}

		let error = prediction - target;
		let absolute_error = error.abs();
		absolute_errors.push(absolute_error);
		squared_error_sum += error * error;
		signed_error_sum += error;
		target_sum += target;
		pairs.push((target, error));
	}

	let sample_count = pairs.len();
	if sample_count == 0 {
		return Err(RegressionMetricsError::EmptyInput);
	}

	absolute_errors.sort_by(f64::total_cmp);
	let count = sample_count as f64;
	let mean_squared_error = squared_error_sum / count;
	let target_mean = target_sum / count;
	let total_sum_squares = pairs
		.iter()
		.map(|(target, _)| (target - target_mean).powi(2))
		.sum::<f64>();
	let r_squared = (total_sum_squares > 0.0)
		.then(|| 1.0 - squared_error_sum / total_sum_squares);

	Ok(RegressionMetrics {
		sample_count,
		mean_squared_error,
		root_mean_squared_error: mean_squared_error.sqrt(),
		mean_absolute_error: absolute_errors.iter().sum::<f64>() / count,
		mean_error: signed_error_sum / count,
		median_absolute_error: percentile(&absolute_errors, 0.5)
			.ok_or(RegressionMetricsError::EmptyInput)?,
		percentile_95_absolute_error: percentile(&absolute_errors, 0.95)
			.ok_or(RegressionMetricsError::EmptyInput)?,
		r_squared,
	})
}

#[expect(
	clippy::cast_possible_truncation,
	clippy::cast_precision_loss,
	clippy::cast_sign_loss,
	reason = "The interpolated position is finite and bounded by the allocated vector length"
)]
fn percentile(
	sorted_values: &[f64],
	quantile: f64,
) -> Option<f64> {
	let last_index = sorted_values.len().checked_sub(1)?;
	let position = last_index as f64 * quantile;
	let lower_index = position.floor() as usize;
	let upper_index = position.ceil() as usize;
	let lower_value = sorted_values.get(lower_index).copied()?;
	let upper_value = sorted_values.get(upper_index).copied()?;
	let fraction = position - lower_index as f64;
	Some(lower_value + fraction * (upper_value - lower_value))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn computes_regression_metrics_and_interpolated_percentiles() {
		let metrics = regression_metrics([(1.0, 2.0), (2.0, 2.0), (3.0, 5.0)])
			.expect("finite non-empty data should produce metrics");

		assert_eq!(metrics.sample_count, 3);
		assert!((metrics.mean_squared_error - 5.0 / 3.0).abs() < 1e-12);
		assert!(
			(metrics.root_mean_squared_error - (5_f64 / 3.0).sqrt()).abs()
				< 1e-12
		);
		assert!((metrics.mean_absolute_error - 1.0).abs() < 1e-12);
		assert!((metrics.mean_error - 1.0).abs() < 1e-12);
		assert!((metrics.median_absolute_error - 1.0).abs() < 1e-12);
		assert!((metrics.percentile_95_absolute_error - 1.9).abs() < 1e-12);
		assert!(
			matches!(metrics.r_squared, Some(value) if (value + 1.5).abs() < 1e-12)
		);
	}

	#[test]
	fn returns_no_r_squared_for_constant_targets() {
		let metrics = regression_metrics([(4.0, 4.0), (4.0, 5.0)])
			.expect("finite non-empty data should produce metrics");

		assert_eq!(metrics.r_squared, None);
	}

	#[test]
	fn rejects_empty_input() {
		assert_eq!(
			regression_metrics([]),
			Err(RegressionMetricsError::EmptyInput)
		);
	}

	#[test]
	fn rejects_non_finite_values_with_sample_index() {
		assert_eq!(
			regression_metrics([(1.0, 1.0), (f64::NAN, 2.0)]),
			Err(RegressionMetricsError::NonFiniteValue {
				sample_index: 1
			})
		);
	}

	#[test]
	fn report_json_and_csv_preserve_sample_data_and_escape_metadata() {
		let results = [SampleInference {
			image_path: "dummy/sample.png".to_owned(),
			position: Some("capture, north".to_owned()),
			expected_intensity: 5_f32,
			predicted_intensity: 4.5_f32,
			source_fields: vec![("note", "quoted \"value\"".to_owned())],
		}];
		let metrics = regression_metrics([(5.0, 4.5)])
			.expect("finite non-empty data should produce metrics");
		let report =
			EvaluationReport::new("dummy", "validation", metrics, &results);

		let mut json = Vec::new();
		report
			.write_json(&mut json)
			.expect("report should serialize as JSON");
		let json_value: serde_json::Value =
			serde_json::from_slice(&json).expect("report JSON should parse");
		assert_eq!(
			json_value
				.get("dataset")
				.and_then(serde_json::Value::as_str),
			Some("dummy")
		);
		assert_eq!(
			json_value.get("split").and_then(serde_json::Value::as_str),
			Some("validation")
		);
		let sample = json_value
			.get("samples")
			.and_then(serde_json::Value::as_array)
			.and_then(|samples| samples.first());
		assert_eq!(
			sample
				.and_then(|sample| sample.get("sample_index"))
				.and_then(serde_json::Value::as_u64),
			Some(0)
		);
		assert_eq!(
			sample
				.and_then(|sample| sample.get("residual"))
				.and_then(serde_json::Value::as_f64),
			Some(-0.5)
		);

		let mut csv_data = Vec::new();
		report
			.write_csv(&mut csv_data)
			.expect("sample rows should serialize as CSV");
		let mut reader = csv::Reader::from_reader(csv_data.as_slice());
		let headers =
			reader.headers().expect("CSV headers should exist").clone();
		let record = reader
			.records()
			.next()
			.expect("CSV should contain one sample")
			.expect("sample CSV row should parse");
		let position_index = headers
			.iter()
			.position(|header| header == "position")
			.expect("position column should exist");
		let metadata_index = headers
			.iter()
			.position(|header| header == "source_fields_json")
			.expect("metadata column should exist");
		assert_eq!(record.get(position_index), Some("capture, north"));
		let metadata_text = record
			.get(metadata_index)
			.expect("metadata field should exist");
		let metadata: BTreeMap<String, String> =
			serde_json::from_str(metadata_text)
				.expect("metadata JSON should parse");
		assert_eq!(
			metadata.get("note").map(String::as_str),
			Some("quoted \"value\"")
		);
	}
}
