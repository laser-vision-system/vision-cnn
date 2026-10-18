//! Custom layer module for convolution layers.

#![expect(
	clippy::redundant_field_names,
	reason = "Caused by Burn's `Config` derive"
)]

use burn::{
	nn::{
		BatchNorm,
		BatchNormConfig,
		PaddingConfig2d,
		conv::{
			Conv2d,
			Conv2dConfig,
		},
		pool::{
			MaxPool2d,
			MaxPool2dConfig,
		},
	},
	prelude::*,
	tensor::activation::relu,
};

/// Configuration options for the [`ConvBlock`].
#[derive(Config, Debug)]
pub struct ConvBlockConfig {
	input_channels: usize,
	output_channels: usize,
	#[config(default = 3)]
	kernel_size: usize,
	padding: PaddingConfig2d,
	#[config(default = 0.1)]
	momentum: f64,
	#[config(default = 4)]
	pool_kernel_size: usize,
}

impl ConvBlockConfig {
	/// Construct a [`ConvBlock`] with the selected configuration.
	pub fn init<B>(
		&self,
		device: &B::Device,
	) -> ConvBlock<B>
	where
		B: Backend,
	{
		ConvBlock {
			conv: Conv2dConfig::new(
				[self.input_channels, self.output_channels],
				[self.kernel_size; 2],
			)
			.with_padding(self.padding.clone())
			.init(device),
			batch_norm: BatchNormConfig::new(self.output_channels)
				.with_momentum(self.momentum)
				.init(device),
			max_pool: MaxPool2dConfig::new([self.pool_kernel_size; 2]).init(),
		}
	}
}

/// A chain of a [`Conv2d`], [`BatchNorm`], and [`MaxPool2d`].
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
	conv: Conv2d<B>,
	batch_norm: BatchNorm<B>,
	max_pool: MaxPool2d,
}

impl<B: Backend> ConvBlock<B> {
	/// Runs the forward pass for this layer module.
	pub fn forward(
		&self,
		input: Tensor<B, 4>,
	) -> Tensor<B, 4> {
		let x = input;
		let x = self.conv.forward(x); // [batch_size, c, w, h]
		let x = self.batch_norm.forward(x);
		let x = relu(x);
		self.max_pool.forward(x) // [batch_size, c, w / pool_kernel_size, h / pool_kernel_size]
	}
}
