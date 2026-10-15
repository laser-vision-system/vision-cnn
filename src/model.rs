pub mod conv_block;
pub mod global_avg_pool;

use burn::{
	nn::{
		Dropout,
		DropoutConfig,
		Linear,
		LinearConfig,
		PaddingConfig2d,
		loss::{
			MseLoss,
			Reduction,
		},
	},
	prelude::*,
	tensor::backend::AutodiffBackend,
	train::{
		InferenceStep,
		RegressionOutput,
		TrainOutput,
		TrainStep,
	},
};

use self::{
	conv_block::{
		ConvBlock,
		ConvBlockConfig,
	},
	global_avg_pool::GlobalAvgPooling2d,
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
	pub fn init<B>(
		&self,
		device: &B::Device,
	) -> LaserModel<B>
	where
		B: Backend,
	{
		LaserModel {
			read_head: ConvBlockConfig::new(3, 16, PaddingConfig2d::Same)
				.with_momentum(self.batch_norm_momentum)
				.init(device),
			detail_extraction: ConvBlockConfig::new(
				16,
				8,
				PaddingConfig2d::Same,
			)
			.with_momentum(self.batch_norm_momentum)
			.init(device),
			global_avg_pool: GlobalAvgPooling2d,
			dropout: DropoutConfig::new(self.dropout).init(),
			prediction: LinearConfig::new(8, 1).init(device),
		}
	}
}

#[derive(Module, Debug)]
pub struct LaserModel<B: Backend> {
	read_head: ConvBlock<B>,
	detail_extraction: ConvBlock<B>,
	global_avg_pool: GlobalAvgPooling2d,
	dropout: Dropout,
	prediction: Linear<B>,
}

impl<B: Backend> LaserModel<B> {
	/// A single forward pass on the model.
	///
	/// # Shapes
	///   - Images `[batch_size, channels, height, width]`
	///   - Output `[batch_size, 1]`
	pub fn forward(
		&self,
		input: Tensor<B, 4>,
	) -> Tensor<B, 2> {
		let [_, channels, ..] = input.dims();
		debug_assert_eq!(channels, 3, "expected an RGB tensor");

		let x = input; // [batch_size, 3, 640, 480]
		let x = self.read_head.forward(x); // [batch_size, 16, 160, 120]
		let x = self.detail_extraction.forward(x); // [batch_size, 8, 40, 30]
		let x = self.global_avg_pool.forward(x); // [batch_size, 8]
		let x = self.dropout.forward(x);
		self.prediction.forward(x) // [batch_size, 1]
	}

	pub fn forward_step(
		&self,
		batch: LaserBatch<B>,
	) -> RegressionOutput<B> {
		let targets: Tensor<B, 2> = batch.targets.unsqueeze_dim(1);
		let logits = self.forward(batch.images);
		let loss = MseLoss::new().forward(
			logits.clone(),
			targets.clone(),
			Reduction::Auto,
		);

		RegressionOutput::new(loss, logits, targets)
	}
}

impl<B: AutodiffBackend> TrainStep for LaserModel<B> {
	type Input = LaserBatch<B>;
	type Output = RegressionOutput<B>;

	fn step(
		&self,
		batch: Self::Input,
	) -> TrainOutput<Self::Output> {
		let item = self.forward_step(batch);
		TrainOutput::new(self, item.loss.backward(), item)
	}
}

impl<B: Backend> InferenceStep for LaserModel<B> {
	type Input = LaserBatch<B>;
	type Output = RegressionOutput<B>;

	fn step(
		&self,
		batch: Self::Input,
	) -> Self::Output {
		self.forward_step(batch)
	}
}

#[cfg(test)]
mod tests {
	use burn::{
		backend::Flex,
		tensor::Tensor,
	};

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
