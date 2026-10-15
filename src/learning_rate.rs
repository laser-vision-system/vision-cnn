use burn::{
	optim::lr_scheduler::LrScheduler,
	tensor::backend::Backend,
};

/// Inverse-time decay scheduler mirroring the Python prototype's
/// `InverseTimeDecay` behaviour.
#[derive(Clone, Debug)]
pub struct InverseTimeDecayLrScheduler {
	initial_lr: f64,
	decay_steps: usize,
	decay_rate: f64,
	step_count: usize,
}

impl InverseTimeDecayLrScheduler {
	/// Create a new inverse-time decay scheduler.
	pub fn new(
		initial_lr: f64,
		decay_steps: usize,
		decay_rate: f64,
	) -> Self {
		Self {
			initial_lr,
			decay_steps,
			decay_rate,
			step_count: 0,
		}
	}

	#[expect(
		clippy::cast_precision_loss,
		reason = "usize is needed to account for long training runs"
	)]
	fn learning_rate_at(
		&self,
		step: usize,
	) -> f64 {
		if self.decay_steps == 0 {
			return self.initial_lr;
		}

		let divisor =
			1f64 + self.decay_rate * step as f64 / self.decay_steps as f64;
		self.initial_lr / divisor
	}
}

impl LrScheduler for InverseTimeDecayLrScheduler {
	type Record<B: Backend> = usize;

	fn step(&mut self) -> f64 {
		let lr = self.learning_rate_at(self.step_count);
		self.step_count += 1;
		lr
	}

	fn to_record<B>(&self) -> Self::Record<B>
	where
		B: Backend,
	{
		self.step_count
	}

	fn load_record<B>(
		mut self,
		record: Self::Record<B>,
	) -> Self
	where
		B: Backend,
	{
		self.step_count = record;
		self
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn inverse_time_decay_schedule_matches_expected_curve() {
		let mut scheduler = InverseTimeDecayLrScheduler::new(0.001, 10, 0.2);
		let expected = [
			0.001_f64,
			0.000_980_392_156_862_745_f64,
			0.000_961_538_461_538_461_6_f64,
			0.000_943_396_226_415_094_4_f64,
			0.000_925_925_925_925_925_9_f64,
			0.000_909_090_909_090_909_1_f64,
		];

		for (step, expected_lr) in expected.iter().enumerate() {
			let actual = scheduler.step();
			assert!(
				(actual - *expected_lr).abs() < 1e-9_f64,
				"step {step}: {actual} != {expected_lr}"
			);
		}
	}

	#[test]
	fn zero_decay_steps_does_not_divide_by_zero() {
		let mut scheduler = InverseTimeDecayLrScheduler::new(0.001, 0, 0.2);
		let step_1 = scheduler.step() - 0.001_f64;
		let step_2 = scheduler.step() - 0.001_f64;
		assert!(step_1.abs() < 1e-9_f64);
		assert!(step_2.abs() < 1e-9_f64);
	}
}
