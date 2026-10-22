//! Inference command modules and workflow dispatch.

mod args;
pub mod evaluate;
mod paths;
mod workflow;

use std::error::Error;

pub use args::InferArgs;

/// Execute sample inference or full-split evaluation.
///
/// # Errors
///
/// Returns an error when paths, model loading, inference, or report writing
/// fail.
pub fn run(args: &InferArgs) -> Result<(), Box<dyn Error>> {
	workflow::run(args)
}
