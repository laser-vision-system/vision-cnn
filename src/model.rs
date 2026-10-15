#![expect(clippy::module_name_repetitions, reason = "Not a problem")]

use burn::{
	nn::{
		BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear,
		LinearConfig, PaddingConfig2d,
		conv::{Conv2d, Conv2dConfig},
		loss::{MseLoss, Reduction},
		pool::{
			AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d,
			MaxPool2dConfig,
		},
	},
	prelude::*,
	tensor::{activation::relu, backend::AutodiffBackend},
	train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};

use crate::data::LaserBatch;

/// Model hyper-parameters for the laser regression CNN.
#[derive(Config, Debug)]
pub struct LaserModelConfig {
	#[config(default = 0.2)]
	pub dropout: f64,
	#[config(default = 0.1)]
	pub batch_norm_momentum: f64,
}

impl LaserModelConfig {
	/// Create a [`LaserModel`] with the configured hyper-parameters.
	pub fn init<B>(&self, device: &B::Device) -> LaserModel<B>
	where
		B: Backend,
	{
		LaserModel {
			read_head: Conv2dConfig::new([3, 16], [3, 3])
				.with_padding(PaddingConfig2d::Same)
				.init(device),
			read_head_norm: BatchNormConfig::new(16)
				.with_momentum(self.batch_norm_momentum)
				.init(device),
			read_head_pool: MaxPool2dConfig::new([4, 4]).init(),
			detail_extraction: Conv2dConfig::new([16, 8], [3, 3])
				.with_padding(PaddingConfig2d::Same)
				.init(device),
			detail_extraction_norm: BatchNormConfig::new(8)
				.with_momentum(self.batch_norm_momentum)
				.init(device),
			detail_extraction_pool: MaxPool2dConfig::new([4, 4]).init(),
			avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
			dropout: DropoutConfig::new(self.dropout).init(),
			prediction: LinearConfig::new(8, 1).init(device),
		}
	}
}

#[derive(Module, Debug)]
pub struct LaserModel<B: Backend> {
	read_head: Conv2d<B>,
	read_head_norm: BatchNorm<B>,
	read_head_pool: MaxPool2d,
	detail_extraction: Conv2d<B>,
	detail_extraction_norm: BatchNorm<B>,
	detail_extraction_pool: MaxPool2d,
	// Burn does not support GlobalAveragePooling2D
	avg_pool: AdaptiveAvgPool2d,
	dropout: Dropout,
	prediction: Linear<B>,
}

impl<B: Backend> LaserModel<B> {
	/// A single forward pass on the model.
	///
	/// # Shapes
	///   - Images \[`batch_size`, channels, height, width\]
	///   - Output \[`batch_size`, 1\]
	pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
		let [batch_size, channels, _, _] = images.dims();
		debug_assert_eq!(channels, 3, "expected an RGB tensor");

		let x = images;
		let x = self.read_head.forward(x); // [batch_size, 16, 640, 480]
		let x = self.read_head_norm.forward(x);
		let x = relu(x);
		let x = self.read_head_pool.forward(x); // [batch_size, 16, 160, 120]
		let x = self.detail_extraction.forward(x); // [batch_size, 8, 160, 120]
		let x = self.detail_extraction_norm.forward(x);
		let x = relu(x);
		let x = self.detail_extraction_pool.forward(x); // [batch_size, 8, 40, 30]
		// These two layers function as the `GlobalAveragePooling2D` layer and
		// drop the shape to two dimensions.
		let x = self.avg_pool.forward(x); // [batch_size, 8, 1, 1]
		let x = x.reshape([batch_size, 8]); // [batch_size, 8]
		let x = self.dropout.forward(x);
		self.prediction.forward(x) // [batch_size, 1]
	}

	pub fn forward_step(&self, batch: LaserBatch<B>) -> RegressionOutput<B> {
		let targets: Tensor<B, 2> = batch.targets.unsqueeze_dim(1);
		let output = self.forward(batch.images);
		let loss = MseLoss::new().forward(
			output.clone(),
			targets.clone(),
			Reduction::Auto,
		);

		RegressionOutput::new(loss, output, targets)
	}
}

impl<B: AutodiffBackend> TrainStep for LaserModel<B> {
	type Input = LaserBatch<B>;
	type Output = RegressionOutput<B>;

	fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
		let item = self.forward_step(batch);
		TrainOutput::new(self, item.loss.backward(), item)
	}
}

impl<B: Backend> InferenceStep for LaserModel<B> {
	type Input = LaserBatch<B>;
	type Output = RegressionOutput<B>;

	fn step(&self, batch: Self::Input) -> Self::Output {
		self.forward_step(batch)
	}
}

#[cfg(test)]
mod tests {
	use burn::{backend::Flex, tensor::Tensor};

	use super::*;

	type Backend = Flex;
	type Device = burn::prelude::Device<Backend>;

	#[test]
	fn model_config_exposes_dropout_configuration() {
		let config = LaserModelConfig::new().with_dropout(0.25);
		assert!((config.dropout - 0.25).abs() < f64::EPSILON);
	}

	#[test]
	fn forward_keeps_batch_shape_for_normalized_inputs() {
		let device = Device::default();
		let model = LaserModelConfig::new()
			.with_dropout(0.1)
			.init::<Backend>(&device);
		let images = Tensor::<Backend, 4>::ones(
			[2_i32, 3_i32, 480_i32, 640_i32],
			&device,
		);
		let output = model.forward(images);

		assert_eq!(output.dims(), [2, 1]);
	}
}
