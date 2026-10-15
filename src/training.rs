#![expect(
	clippy::expect_used,
	clippy::module_name_repetitions,
	reason = "Not a problem"
)]

use std::{fs, path::Path};

use burn::{
	data::{dataloader::DataLoaderBuilder, dataset::Dataset},
	optim::AdamConfig,
	prelude::*,
	record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
	tensor::backend::AutodiffBackend,
	train::{
		Learner, MetricEarlyStoppingStrategy, StoppingCondition,
		SupervisedTraining,
		metric::{
			CpuMemory, CudaMetric, LossMetric,
			store::{Aggregate, Direction, Split},
		},
	},
};
use dataset_loader::item::NumericLike;

use crate::{
	data::{
		AugmentedLaserBatcher, ImageStep, IntensityValType, LoadedLaserItem,
	},
	learning_rate::InverseTimeDecayLrScheduler,
	model::LaserModelConfig,
	persistence,
};

/// Configuration for training hyper-parameters.
#[derive(Config, Debug)]
pub struct TrainingConfig {
	pub model: LaserModelConfig,
	pub optimiser: AdamConfig,
	#[config(default = 10)]
	pub num_epochs: usize,
	#[config(default = 16)]
	pub batch_size: usize,
	#[config(default = 4)]
	pub num_workers: usize,
	#[config(default = 42)]
	pub seed: u64,
	#[config(default = 1e-3)]
	pub learning_rate: f64,
	#[config(default = 0.2)]
	pub decay_rate: f64,
	#[config(default = 0)]
	pub decay_steps: usize,
	#[config(default = 2)]
	pub early_stopping_patience: usize,
}

fn create_artefact_dir(artefact_dir: &str) {
	_ = fs::create_dir_all(artefact_dir);
}

fn copy_checkpoints(source_dir: &str, output_dir: &str) {
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

/// Find the latest model checkpoint epoch in an artifact directory.
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

#[expect(clippy::integer_division, reason = "Accounted for")]
fn steps_per_epoch(dataset_len: usize, batch_size: usize) -> usize {
	if dataset_len == 0 || batch_size == 0 {
		return 1;
	}

	let full_batches = dataset_len / batch_size;
	let partial_batch = usize::from(!dataset_len.is_multiple_of(batch_size));
	let steps = full_batches + partial_batch;
	if steps == 0 { 1 } else { steps }
}

/// Execute the training process.
pub fn train<B, D, T, AT, AV>(
	artefact_dir: &str,
	config: &TrainingConfig,
	device: &B::Device,
	train_split: D,
	val_split: D,
	train_augmentation: AT,
	valid_augmentation: AV,
	resume: Option<(&str, usize)>,
) where
	B: AutodiffBackend,
	D: Dataset<LoadedLaserItem<T>> + 'static,
	T: NumericLike + Into<IntensityValType> + 'static,
	AT: ImageStep + 'static,
	AV: ImageStep + 'static,
{
	create_artefact_dir(artefact_dir);
	if let Some((resume_dir, _)) = resume {
		copy_checkpoints(resume_dir, artefact_dir);
	}
	config
		.save(format!("{artefact_dir}/config.json"))
		.expect("Config should be saved successfully");

	B::seed(device, config.seed);

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

	let valid_loss_metric = LossMetric::new();
	let mut training = SupervisedTraining::new(
		artefact_dir,
		dataloader_train,
		dataloader_valid,
	)
	.metric_train_numeric(LossMetric::new())
	.metric_train(CudaMetric::new())
	.metric_train(CpuMemory::new())
	.metric_valid_numeric(valid_loss_metric.clone())
	.metric_valid(CudaMetric::new())
	.metric_valid(CpuMemory::new())
	.early_stopping(MetricEarlyStoppingStrategy::new(
		&valid_loss_metric,
		Aggregate::Mean,
		Direction::Lowest,
		Split::Valid,
		StoppingCondition::NoImprovementSince {
			n_epochs: config.early_stopping_patience,
		},
	))
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
