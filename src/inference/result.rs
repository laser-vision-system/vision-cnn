//! A data structure for the results of inference.

use std::fmt::{
	self,
	Display,
};

use burn::tensor::Element;

/// The metadata and inferred intensity for one dataset sample.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleInference<T>
where
	T: Element,
{
	/// Canonical image path identifying this sample on the current machine.
	pub image_path: String,
	/// Position metadata, when the source dataset provides it.
	pub position: Option<String>,
	/// Dataset intensity for this sample.
	pub expected_intensity: T,
	/// Model-predicted intensity.
	pub predicted_intensity: T,
	/// Named source-specific fields, excluding intensity and position.
	pub source_fields: Vec<(&'static str, String)>,
}

impl<T> SampleInference<T>
where
	T: Element,
{
	fn row(
		&self,
		field_names: &[&str],
	) -> Vec<String> {
		[
			self.position.clone().unwrap_or_default(),
			format!("{:.3}", self.expected_intensity),
			format!("{:.3}", self.predicted_intensity),
		]
		.into_iter()
		.chain(field_names.iter().map(|name| {
			self.source_fields
				.iter()
				.find(|(field_name, _)| field_name == name)
				.map_or_else(String::new, |(_, value)| value.clone())
		}))
		.collect()
	}
}

impl<T> Display for SampleInference<T>
where
	T: Element,
{
	fn fmt(
		&self,
		f: &mut fmt::Formatter<'_>,
	) -> fmt::Result {
		if let Some(position) = &self.position {
			write!(f, "Position: {position}")?;
		}
		write!(
			f,
			": expected {}, predicted {:.3}",
			self.expected_intensity, self.predicted_intensity
		)?;
		for (name, value) in &self.source_fields {
			write!(f, "; {name}: {value}")?;
		}
		Ok(())
	}
}

/// A collection of sample inference results, formatted as a table.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceResults<T>(pub Vec<SampleInference<T>>)
where
	T: Element;

impl<T> InferenceResults<T>
where
	T: Element,
{
	fn source_field_names(&self) -> Vec<&'static str> {
		let mut names = Vec::new();
		for result in &self.0 {
			for (name, _) in &result.source_fields {
				if !names.contains(name) {
					names.push(*name);
				}
			}
		}
		names
	}

	fn rows(
		&self,
		field_names: &[&str],
	) -> Vec<Vec<String>> {
		self.0
			.iter()
			.map(|result| result.row(field_names))
			.collect()
	}
}

impl<T> Display for InferenceResults<T>
where
	T: Element,
{
	fn fmt(
		&self,
		f: &mut fmt::Formatter<'_>,
	) -> fmt::Result {
		let field_names = self.source_field_names();
		let headers = table_headers(&field_names);
		let rows = self.rows(&field_names);
		let widths = column_widths(&headers, &rows);

		write_cells(f, &headers, &widths, true)?;
		for row in &rows {
			write_cells(f, row, &widths, false)?;
		}
		Ok(())
	}
}

fn table_headers(field_names: &[&str]) -> Vec<String> {
	["Position", "Expected", "Predicted"]
		.into_iter()
		.map(str::to_owned)
		.chain(field_names.iter().map(ToString::to_string))
		.collect()
}

fn column_widths(
	headers: &[String],
	rows: &[Vec<String>],
) -> Vec<usize> {
	let mut widths = headers.iter().map(String::len).collect::<Vec<_>>();
	for row in rows {
		for (width, cell) in widths.iter_mut().zip(row) {
			*width = (*width).max(cell.len());
		}
	}
	widths
}

fn write_cells(
	formatter: &mut fmt::Formatter<'_>,
	cells: &[impl Display],
	widths: &[usize],
	left_aligned: bool,
) -> fmt::Result {
	for (index, (cell, width)) in cells.iter().zip(widths).enumerate() {
		if index > 0 {
			write!(formatter, "  ")?;
		}
		if left_aligned || index == 0 {
			write!(formatter, "{cell:<width$}")?;
		} else {
			write!(formatter, "{cell:>width$}")?;
		}
	}
	writeln!(formatter)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn displays_result_as_an_aligned_table_row() {
		let result = SampleInference {
			image_path: "dummy/sample.png".to_owned(),
			position: Some("rearwards_close_outer_0".to_owned()),
			source_fields: vec![("Gamma", "140".to_owned())],
			expected_intensity: 3.5_f32,
			predicted_intensity: 3.25,
		};

		assert!(result.to_string().contains("rearwards_close_outer_0"));
		assert!(result.to_string().contains("expected 3.5, predicted 3.250"));
	}

	#[test]
	fn retains_sample_and_intensity_values() {
		let result = SampleInference {
			image_path: "dummy/sample.png".to_owned(),
			position: None,
			source_fields: vec![],
			expected_intensity: 12_f32,
			predicted_intensity: 11.75_f32,
		};

		assert_eq!(result.position, None);
		assert_eq!(result.source_fields, [] as [(&str, String); 0]);
		assert!((result.expected_intensity - 12_f32).abs() < f32::EPSILON);
		assert!((result.predicted_intensity - 11.75_f32).abs() < f32::EPSILON);
	}

	#[test]
	fn results_display_a_table_with_source_metadata() {
		let results = InferenceResults(vec![SampleInference {
			image_path: "dummy/sample.png".to_owned(),
			position: Some("rearwards_close_outer_0".to_owned()),
			source_fields: vec![
				("Saturation", "64".to_owned()),
				("ExposureAbsolute", "405".to_owned()),
				("Gamma", "270".to_owned()),
			],
			expected_intensity: 3.2_f32,
			predicted_intensity: 3.15,
		}]);
		let output = results.to_string();

		assert!(output.contains("Position"));
		assert!(output.contains("Saturation"));
		assert!(output.contains("ExposureAbsolute"));
		assert!(output.contains("Gamma"));
		assert!(output.contains("rearwards_close_outer_0"));
		assert!(!output.contains("No position data"));
		assert_eq!(output.matches("rearwards_close_outer_0").count(), 1);
		assert!(output.lines().any(|line| {
			line.contains("3.2")
				&& line.contains("64")
				&& line.contains("405")
				&& line.contains("270")
				&& line.contains("3.150")
		}));
	}
}
