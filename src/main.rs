//! Command-line entry point for the `vision-cnn` application.

use std::error::Error;

mod commands;

fn main() -> Result<(), Box<dyn Error>> {
	commands::run()
}
