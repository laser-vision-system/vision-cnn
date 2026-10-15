use std::error::Error;

use cnn::commands::run;

fn main() -> Result<(), Box<dyn Error>> {
	run()
}
