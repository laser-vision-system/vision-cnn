#![expect(
	clippy::print_stdout,
	reason = "Evaluation reports its results in the terminal"
)]

use std::{
	error::Error,
	fs::{
		self,
		File,
	},
	io::{
		Error as IoError,
		ErrorKind,
	},
	num::NonZeroUsize,
	path::{
		Path,
		PathBuf,
	},
};

use burn::{
	data::dataset::Dataset,
	prelude::Backend,
	tensor::Element,
};

use crate::{
	data::{
		IntensityValType,
		LoadedLaserItem,
	},
	evaluation::{
		EvaluationReport,
		RegressionMetrics,
		regression_metrics,
	},
	inference,
};

pub(super) fn infer_samples<B, T, D>(
	model_dir: &Path,
	num_samples: usize,
	dataset: &D,
	device: &B::Device,
) -> Result<(), Box<dyn Error>>
where
	B: Backend,
	T: Element + Into<IntensityValType>,
	D: Dataset<LoadedLaserItem<T>>,
{
	let model_dir = model_dir.to_str().ok_or_else(|| {
		IoError::new(ErrorKind::InvalidInput, "model path is not UTF-8")
	})?;
	let items: Vec<_> = dataset.iter().take(num_samples).collect();
	let results = inference::infer::<B, T>(model_dir, device, items);
	print_results(&results);
	Ok(())
}

pub(super) fn evaluate_all<B, T, D>(
	model_dir: &Path,
	batch_size: NonZeroUsize,
	dataset_name: &str,
	split_name: &str,
	report_dir: &Path,
	dataset: &D,
	device: &B::Device,
) -> Result<(), Box<dyn Error>>
where
	B: Backend,
	T: Element + Into<IntensityValType>,
	D: Dataset<LoadedLaserItem<T>>,
{
	let model_dir = model_dir.to_str().ok_or_else(|| {
		IoError::new(ErrorKind::InvalidInput, "model path is not UTF-8")
	})?;
	let results = inference::infer_dataset::<B, T, D>(
		model_dir, device, dataset, batch_size,
	)?;
	let metrics = regression_metrics(results.0.iter().map(|sample| {
		(
			f64::from(sample.expected_intensity),
			f64::from(sample.predicted_intensity),
		)
	}))?;
	let report =
		EvaluationReport::new(dataset_name, split_name, metrics, &results.0);
	let (summary_path, samples_path) = write_report(report_dir, &report)?;

	print_metrics(&metrics);
	print_report_paths(&summary_path, &samples_path);
	Ok(())
}

fn write_report(
	report_dir: &Path,
	report: &EvaluationReport,
) -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
	fs::create_dir_all(report_dir)?;
	let summary_path = report_dir.join("metrics.json");
	let samples_path = report_dir.join("samples.csv");
	report.write_json(File::create(&summary_path)?)?;
	report.write_csv(File::create(&samples_path)?)?;
	Ok((summary_path, samples_path))
}

fn print_metrics(metrics: &RegressionMetrics) {
	println!("Evaluation samples: {}", metrics.sample_count);
	println!("MSE: {:.6}", metrics.mean_squared_error);
	println!("RMSE: {:.6}", metrics.root_mean_squared_error);
	println!("MAE: {:.6}", metrics.mean_absolute_error);
	println!(
		"Mean error (prediction - target): {:.6}",
		metrics.mean_error
	);
	println!(
		"Median absolute error: {:.6}",
		metrics.median_absolute_error
	);
	println!(
		"95th percentile absolute error: {:.6}",
		metrics.percentile_95_absolute_error
	);
	match metrics.r_squared {
		Some(value) => println!("R-squared: {value:.6}"),
		None => println!("R-squared: undefined (constant targets)"),
	}
}

fn print_results<T>(results: &inference::result::InferenceResults<T>)
where
	T: Element,
{
	println!("{results}");
}

fn print_report_paths(
	summary_path: &Path,
	samples_path: &Path,
) {
	println!("Summary: {}", summary_path.display());
	println!("Samples: {}", samples_path.display());
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
	use crate::{
		evaluation::regression_metrics,
		inference::result::SampleInference,
	};

	static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

	#[test]
	fn report_writer_creates_summary_and_sample_files() {
		let unique = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
		let report_dir = env::temp_dir()
			.join(format!("cnn-evaluation-report-{}-{unique}", process::id()));
		let metrics = regression_metrics([(2.0, 3.0)])
			.expect("finite non-empty data should produce metrics");
		let samples = [SampleInference {
			image_path: "dummy/sample.png".to_owned(),
			position: Some("sample".to_owned()),
			expected_intensity: 2_f32,
			predicted_intensity: 3_f32,
			source_fields: Vec::new(),
		}];
		let report = EvaluationReport::new("dummy", "test", metrics, &samples);

		let paths = write_report(&report_dir, &report)
			.expect("report files should be written");
		let summary = fs::read_to_string(&paths.0)
			.expect("summary JSON should be readable");
		let rows = fs::read_to_string(&paths.1)
			.expect("sample CSV should be readable");
		assert!(summary.contains("\"split\": \"test\""));
		assert!(rows.contains("dummy/sample.png"));
		fs::remove_dir_all(report_dir)
			.expect("temporary report directory should be removed");
	}
}
