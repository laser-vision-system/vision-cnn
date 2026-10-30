//! Command-line entry point for the `vision-cnn-cli` application.

fn main() -> Result<(), Box<dyn std::error::Error>> {
	vision_cnn_cli::run()
}
