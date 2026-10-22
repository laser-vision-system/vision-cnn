//! Training configuration, checkpoint discovery, and model optimization.

#![expect(clippy::expect_used, reason = "Not a problem")]
#![expect(
	clippy::redundant_field_names,
	reason = "Caused by Burn's `Config` derive"
)]

use std::{
	fs,
	path::Path,
};

use burn::{
	data::{
		dataloader::DataLoaderBuilder,
		dataset::Dataset as _,
	},
	optim::AdamConfig,
	prelude::*,
	record::{
		FullPrecisionSettings,
		NamedMpkGzFileRecorder,
	},
	tensor::backend::AutodiffBackend,
	train::{
		Learner,
		MetricEarlyStoppingStrategy,
		StoppingCondition,
		SupervisedTraining,
		metric::{
			CpuMemory,
			CpuUse,
			CudaMetric,
			IterationSpeedMetric,
			LearningRateMetric,
			LossMetric,
			store::{
				Aggregate,
				Direction,
				Split,
			},
		},
	},
};
use dataset_loader::augmentation::ImageStep;

use crate::{
	data::{
		AugmentedLaserBatcher,
		DatasetSplits,
	},
	learning_rate::InverseTimeDecayLrScheduler,
	model::LaserModelConfig,
	persistence,
};

/// Default value for the `num_epochs` training hyper-parameter.
pub const NUM_EPOCHS_DEFAULT: usize = 2;
/// Default value for the `batch_size` training hyper-parameter.
pub const BATCH_SIZE_DEFAULT: usize = 8;
/// Default value for the `num_workers` training hyper-parameter.
pub const NUM_WORKERS_DEFAULT: usize = 4;
/// Default value for the `seed` training hyper-parameter.
pub const SEED_DEFAULT: u64 = 42;
/// Default value for the `learning_rate` training hyper-parameter.
pub const LEARNING_RATE_DEFAULT: f64 = 1e-2;
/// Default value for the `decay_rate` training hyper-parameter.
pub const DECAY_RATE_DEFAULT: f64 = 0.2;
/// Default value for the `decay_steps` training hyper-parameter.
pub const DECAY_STEPS_DEFAULT: usize = 0;
/// Default value for the `early_stopping_patience` training hyper-parameter.
pub const EARLY_STOPPING_PATIENCE_DEFAULT: usize = 2;

/// Configuration for training hyper-parameters.
#[derive(Config, Debug)]
pub struct TrainingConfig {
	/// Model architecture and regularisation configuration.
	pub model: LaserModelConfig,
	/// Optimiser configuration.
	pub optimiser: AdamConfig,
	/// Maximum number of training epochs.
	#[config(default = "NUM_EPOCHS_DEFAULT")]
	pub num_epochs: usize,
	/// Number of samples per training and validation batch.
	#[config(default = "BATCH_SIZE_DEFAULT")]
	pub batch_size: usize,
	/// Number of data-loading workers.
	#[config(default = "NUM_WORKERS_DEFAULT")]
	pub num_workers: usize,
	/// Seed used for backend initialization and data ordering.
	#[config(default = "SEED_DEFAULT")]
	pub seed: u64,
	/// Initial optimiser learning rate.
	#[config(default = "LEARNING_RATE_DEFAULT")]
	pub learning_rate: f64,
	/// Inverse-time learning-rate decay factor.
	#[config(default = "DECAY_RATE_DEFAULT")]
	pub decay_rate: f64,
	/// Steps between learning-rate decay intervals; zero derives this from the
	/// training split and batch size.
	#[config(default = "DECAY_STEPS_DEFAULT")]
	pub decay_steps: usize,
	/// Number of validation epochs without improvement before stopping.
	#[config(default = "EARLY_STOPPING_PATIENCE_DEFAULT")]
	pub early_stopping_patience: usize,
}

/// Find the greatest checkpoint epoch in an artifact directory.
///
/// Returns `None` when the checkpoint directory cannot be read or contains no
/// filename with a parseable `model-<epoch>` prefix.
pub fn latest_checkpoint_epoch(artefact_dir: &str) -> Option<usize> {
	fs::read_dir(Path::new(artefact_dir).join("checkpoint"))
		.ok()?
		.filter_map(Result::ok)
		.filter_map(|entry| {
			let filename = entry.file_name();
			let filename = filename.to_str()?.strip_prefix("model-")?;
			let epoch: String =
				filename.chars().take_while(char::is_ascii_digit).collect();
			epoch.parse().ok()
		})
		.max()
}

/// Execute the training process.
///
/// # Panics
///
/// - If the config could not be saved.
/// - If the model could not be exported.
/// - If a requested resume checkpoint cannot be copied.
#[expect(clippy::impl_trait_in_params, reason = "Hide generic from turbofish")]
pub fn train<B>(
	artefact_dir: &str,
	config: &TrainingConfig,
	device: &B::Device,
	splits: DatasetSplits<impl Backend<FloatElem = B::FloatElem>>,
	train_augmentation: impl ImageStep + 'static,
	valid_augmentation: impl ImageStep + 'static,
	resume: Option<(&str, usize)>,
) where
	B: AutodiffBackend,
{
	create_artefact_dir(artefact_dir);
	if let Some((resume_dir, _)) = resume {
		copy_checkpoints(resume_dir, artefact_dir);
	}
	config
		.save(format!("{artefact_dir}/config.json"))
		.expect("Config should be saved successfully");

	B::seed(device, config.seed);

	let train_split = splits.training;
	let val_split = splits.validation;

	let train_steps_per_epoch =
		steps_per_epoch(train_split.len(), config.batch_size);
	let decay_steps = if config.decay_steps == 0 {
		train_steps_per_epoch
	} else {
		config.decay_steps
	};

	let batcher_train =
		AugmentedLaserBatcher::new(train_augmentation, config.seed);

	let dataloader_train = DataLoaderBuilder::new(batcher_train)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(train_split);

	let batcher_valid =
		AugmentedLaserBatcher::new(valid_augmentation, config.seed);

	let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(val_split);

	let loss_metric = LossMetric::new();
	let mut training = SupervisedTraining::new(
		artefact_dir,
		dataloader_train,
		dataloader_valid,
	)
	.early_stopping(MetricEarlyStoppingStrategy::new(
		&loss_metric,
		Aggregate::Mean,
		Direction::Lowest,
		Split::Valid,
		StoppingCondition::NoImprovementSince {
			n_epochs: config.early_stopping_patience,
		},
	))
	// The metrics will be ordered the same as they are added.
	.metrics((loss_metric,))
	.metric_train_numeric(LearningRateMetric::new())
	.metrics((IterationSpeedMetric::new(), CpuUse::new(), CpuMemory::new()))
	.metrics_text((CudaMetric::new(),))
	.with_file_checkpointer(
		NamedMpkGzFileRecorder::<FullPrecisionSettings>::new(),
	)
	.num_epochs(resume.map_or(config.num_epochs, |(_, epoch)| {
		epoch.saturating_add(config.num_epochs)
	}))
	.summary();

	if let Some((_, epoch)) = resume {
		training = training.checkpoint(epoch);
	}

	let model = config.model.init::<B>(device);
	let scheduler = InverseTimeDecayLrScheduler::new(
		config.learning_rate,
		decay_steps,
		config.decay_rate,
	);
	let result = training.launch(Learner::new(
		model,
		config.optimiser.init(),
		scheduler,
	));

	persistence::export_model(
		&result.model,
		Path::new(artefact_dir).join("model.bpk"),
	)
	.expect("Training model should be exported successfully");
}

fn create_artefact_dir(artefact_dir: &str) {
	_ = fs::create_dir_all(artefact_dir);
}

fn copy_checkpoints(
	source_dir: &str,
	output_dir: &str,
) {
	let source = Path::new(source_dir).join("checkpoint");
	let destination = Path::new(output_dir).join("checkpoint");
	fs::create_dir_all(&destination)
		.expect("output checkpoint directory should be created");
	for entry in
		fs::read_dir(source).expect("resume checkpoint directory should exist")
	{
		let entry = entry.expect("checkpoint entry should be readable");
		let file_type = entry
			.file_type()
			.expect("checkpoint file type should be readable");
		#[expect(
			clippy::filetype_is_file,
			reason = "Checkpoint records must be copied only when they are regular files"
		)]
		if file_type.is_file() {
			fs::copy(entry.path(), destination.join(entry.file_name()))
				.expect("resume checkpoint should be copied to output");
		}
	}
}

#[expect(clippy::integer_division, reason = "Accounted for")]
fn steps_per_epoch(
	dataset_len: usize,
	batch_size: usize,
) -> usize {
	if dataset_len == 0 || batch_size == 0 {
		return 1;
	}

	let full_batches = dataset_len / batch_size;
	let partial_batch = usize::from(!dataset_len.is_multiple_of(batch_size));
	let steps = full_batches + partial_batch;
	if steps == 0 { 1 } else { steps }
}
