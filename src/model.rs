//! Types and functions for creating a model and running forward passes on it.

pub mod conv_block;
pub mod global_avg_pool;
pub mod read_head;

use burn::{
	nn::{
		Linear,
		LinearConfig,
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

use self::read_head::{
	ReadHead,
	ReadHeadConfig,
};
use crate::data::LaserBatch;

/// Model hyper-parameters for the laser regression CNN.
#[derive(Config, Debug)]
pub struct LaserModelConfig {
	/// Configuration for the stack that converts images to latent space.
	#[config(default = "ReadHeadConfig::new()")]
	pub read_head_config: ReadHeadConfig,
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
		let start_channels = self.read_head_config.start_channels;
		let double = start_channels * 2;
		let quad = start_channels * 4;
		LaserModel {
			read_head: self.read_head_config.init(device),
			hidden: LinearConfig::new(quad, double).init(device),
			prediction: LinearConfig::new(double, 1).init(device),
		}
	}
}

/// The laser intensity prediction model.
#[derive(Module, Debug)]
pub struct LaserModel<B: Backend> {
	read_head: ReadHead<B>,
	hidden: Linear<B>,
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
		let x = input; // [batch_size, 1, 640, 480]
		let x = self.read_head.forward(x); // [batch_size, 32]
		let x = self.hidden.forward(x); // [batch_size, 32]
		self.prediction.forward(x) // [batch_size, 1]
	}

	/// The training/validation forward pass with loss calculation.
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
	fn forward_keeps_batch_shape_for_normalised_inputs() {
		let device = Device::default();
		let model = LaserModelConfig::new().init::<Backend>(&device);
		let images = Tensor::<Backend, 4>::ones(
			[2_i32, 3_i32, 480_i32, 640_i32],
			&device,
		);
		let output = model.forward(images);

		assert_eq!(output.dims(), [2, 1]);
	}
}
