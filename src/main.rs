use std::error::Error;

use vision_cnn::commands::run;

fn main() -> Result<(), Box<dyn Error>> {
	run()
}
