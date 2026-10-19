//! Reusable data, model, training, inference, and evaluation APIs for laser
//! intensity prediction.

#![cfg_attr(
	test,
	allow(
		clippy::print_stdout,
		clippy::print_stderr,
		clippy::use_debug,
		clippy::panic,
		clippy::unwrap_used,
		clippy::unwrap_in_result,
		clippy::expect_used,
		clippy::indexing_slicing,
		reason = "Tests are allowed to perform all of these operations"
	)
)]

pub mod data;
pub mod evaluation;
pub mod inference;
pub mod learning_rate;
pub mod model;
pub mod persistence;
pub mod plan;
pub mod training;
