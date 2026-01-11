from pathlib import Path
from typing import cast, Optional, Tuple, Dict, Any
import sys
import yaml

import torch
import wandb


from colpali_engine.loss import ColbertPairwiseCELoss
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from datasets import load_dataset, Dataset

from peft import LoraConfig
from PIL import Image
from torch import nn
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param}"
    )


def create_quantization_config(strategy: Optional[str] = "4bit") -> Optional[BitsAndBytesConfig]:
    """
    Create quantization configuration based on the strategy.

    Args:
        strategy: Quantization strategy - "4bit", "8bit", or None

    Returns:
        BitsAndBytesConfig or None
    """
    if strategy is None:
        return None
    elif strategy == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif strategy == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Invalid quantization strategy: {strategy}")


def load_model_and_processor(
    model_name: str,
    quantization_strategy: Optional[str] = "4bit",
    device: str = "auto"
) -> Tuple[ColPali, ColPaliProcessor, str]:
    """
    Load ColPali model and processor with optional quantization.

    Args:
        model_name: Path or name of the pre-trained model
        quantization_strategy: Quantization strategy - "4bit", "8bit", or None
        device: Device to load the model on

    Returns:
        Tuple of (model, processor, device)
    """
    # Automatically set the device
    device = get_torch_device(device)

    if quantization_strategy and device != "cuda:0":
        raise ValueError("Quantization requires a CUDA GPU.")

    # Prepare quantization config
    bnb_config = create_quantization_config(quantization_strategy)


    # Get the LoRA config from the pretrained model
    lora_config = LoraConfig.from_pretrained(model_name)

    # Load the model with the loaded pre-trained adapter
    model = cast(
        ColPali,
        ColPali.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    )

    if not model.active_adapters():
        raise ValueError("No adapter found in the model.")

    # The LoRA weights are frozen by default. We need to unfreeze them to fine-tune the model.
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    print_trainable_parameters(model)

    if lora_config.base_model_name_or_path is None:
        raise ValueError("Base model name or path is required in the LoRA config.")

    processor = cast(
        ColPaliProcessor,
        ColPaliProcessor.from_pretrained(model_name),
    )

    return model, processor, device


def load_training_data(
    data_path: str,
    test_size: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Load and split training data.

    Args:
        data_path: Path to the JSON data file
        test_size: Proportion of data to use for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    ds = load_dataset(
        "json",
        data_files=data_path,
        split="train",
    )

    ds = ds.train_test_split(test_size=test_size, seed=seed)
    train_ds = ds["train"]
    test_ds = ds["test"]

    return train_ds, test_ds


def create_collate_fn(processor: ColPaliProcessor, model: ColPali):
    """
    Create a collate function for the DataLoader.

    Args:
        processor: ColPali processor
        model: ColPali model

    Returns:
        Collate function
    """
    def collate_fn(examples):
        texts = []
        images = []

        for example in examples:
            texts.append(example["query"])
            if "image" in example:
                images.append(example["image"].convert("RGB"))
            elif "image_url" in example:
                img = Image.open(example["image_url"]).convert("RGB")
                images.append(img)
            else:
                raise ValueError("No image or image_url field found in the example.")

        # Use the correct processor methods for ColPali
        batch_queries = processor.process_queries(texts).to(model.device)
        batch_images = processor.process_images(images).to(model.device)

        return (batch_queries, batch_images)

    return collate_fn


class ContrastiveTrainer(Trainer):
    """Custom Trainer for contrastive learning with ColPali."""

    def __init__(self, loss_func, debug_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.debug_dir = debug_dir
        self.step_counter = 0

        # Create debug directory if specified
        if self.debug_dir:
            Path(self.debug_dir).mkdir(parents=True, exist_ok=True)

    def compute_loss(self, model, inputs, num_items_in_batch=4, return_outputs=False):
        query_inputs, doc_inputs = inputs
        query_outputs = model(**query_inputs)
        doc_outputs = model(**doc_inputs)
        loss = self.loss_func(query_outputs, doc_outputs)
        print(loss,)

        # Save debug information if debug_dir is specified
        if self.debug_dir:
            self._save_debug_info(model, query_outputs, doc_outputs, loss)

        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def _save_debug_info(self, model, query_outputs, doc_outputs, loss):
        """Save query_outputs, doc_outputs, and loss for debugging."""
        debug_file = Path(self.debug_dir) / f"step_{self.step_counter:06d}.pt"
        for n, p in model.named_parameters():
            if p.requires_grad and "lora" in n and p.grad is not None:
                print("example lora grad norm:", n, p.grad.data.float().norm().item())
                break

        debug_data = {
            'step': self.step_counter,
            'loss': loss.detach().cpu(),
            'query_outputs': query_outputs.detach().cpu(),
            'doc_outputs': doc_outputs.detach().cpu(),
            'query_shape': query_outputs.shape,
            'doc_shape': doc_outputs.shape,
        }

        torch.save(debug_data, debug_file)
        print(f"Saved debug info to {debug_file}")

        self.step_counter += 1

    def prediction_step(self, model, inputs):
        query_inputs, doc_inputs = inputs
        with torch.no_grad():
            query_outputs = model(**query_inputs)
            doc_outputs = model(**doc_inputs)
            loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None


def create_trainer(
    model: ColPali,
    train_dataset: Dataset,
    collate_fn,
    output_dir: str = "core/colpali-finetune/checkpoints",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    logging_steps: int = 20,
    warmup_steps: int = 100,
    save_total_limit: int = 1,
    debug_dir: Optional[str] = None
) -> ContrastiveTrainer:
    """
    Create and configure the ContrastiveTrainer.

    Args:
        model: ColPali model
        train_dataset: Training dataset
        collate_fn: Collate function for data loading
        output_dir: Directory to save checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        logging_steps: Number of steps between logging
        warmup_steps: Number of warmup steps
        save_total_limit: Maximum number of checkpoints to keep
        debug_dir: Directory to save debug information (query_outputs, doc_outputs, loss)

    Returns:
        Configured ContrastiveTrainer
    """
    checkpoints_dir = Path(output_dir)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        save_steps=100,
        save_total_limit=save_total_limit,
        report_to="wandb",
        dataloader_pin_memory=False
    )

    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        loss_func=ColbertPairwiseCELoss(),
        data_collator=collate_fn,
        debug_dir=debug_dir
    )

    trainer.args.remove_unused_columns = False

    return trainer


def run_inference(
    model: ColPali,
    processor: ColPaliProcessor,
    dataset: Dataset,
    n_examples: int = 3
) -> dict:
    """
    Run inference on a subset of the dataset and evaluate predictions.

    Args:
        model: ColPali model
        processor: ColPali processor
        dataset: Dataset to run inference on
        n_examples: Number of examples to evaluate

    Returns:
        Dictionary containing predictions, scores, and accuracy
    """
    # Load images and queries
    if "image" in dataset.column_names:
        images = [dataset[i]["image"].convert("RGB") for i in range(n_examples)]
    elif "image_url" in dataset.column_names:
        images = [Image.open(dataset[i]["image_url"]).convert("RGB") for i in range(n_examples)]
    else:
        raise ValueError("No image or image_url field found in the dataset.")

    queries = [dataset[i]["query"] for i in range(n_examples)]
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    # Run inference
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    predicted_indices = scores.argmax(dim=1)

    # Calculate accuracy
    accuracy = (predicted_indices == torch.arange(len(queries), device=predicted_indices.device)).float().mean().item()

    return {
        "queries": queries,
        "predicted_indices": predicted_indices,
        "scores": scores,
        "accuracy": accuracy
    }


def print_inference_results(results: dict):
    """
    Print inference results in a formatted way.

    Args:
        results: Dictionary containing inference results
    """
    queries = results["queries"]
    predicted_indices = results["predicted_indices"]
    scores = results["scores"]
    accuracy = results["accuracy"]

    print("Query-Image Matching Results:")
    print("=" * 80)

    for i, (query, pred_idx) in enumerate(zip(queries, predicted_indices)):
        print(f"\nQuery {i}: {query}")
        print(f"Predicted Image Index: {pred_idx.item()}, Score: {scores[i, pred_idx]:.4f}")

        # Check if prediction is correct
        is_correct = pred_idx == i
        if is_correct:
            print("Correct prediction!")
        else:
            print(f"Wrong! Should be image {i}")

        print("-" * 80)

    # Print summary
    print(f"\nSummary:")
    print(f"Predicted indices: {predicted_indices.tolist()}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nScore Matrix (Rows=Queries, Cols=Images):")
    print(scores.cpu().numpy())


def main(
    model_name: str = "/tmp2/share_data/models--vidore--colpali-v1.3",
    data_path: str = "data/ques_gen/data/example.json",
    quantization_strategy: Optional[str] = "4bit",
    output_dir: str = "core/colpali-finetune/checkpoints",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    logging_steps: int = 20,
    save_total_limit: int = 1,
    test_size: float = 0.1,
    seed: int = 42,
    run_pre_training_inference: bool = True,
    run_post_training_inference: bool = True,
    n_inference_examples: int = 3,
    debug_dir: Optional[str] = None
):
    """
    Main function to run the ColPali fine-tuning pipeline.

    Args:
        model_name: Path or name of the pre-trained model
        data_path: Path to the training data JSON file
        quantization_strategy: Quantization strategy - "4bit", "8bit", or None
        output_dir: Directory to save checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        test_size: Proportion of data to use for testing
        seed: Random seed for reproducibility
        run_pre_training_inference: Whether to run inference before training
        run_post_training_inference: Whether to run inference after training
        n_inference_examples: Number of examples for inference evaluation
        debug_dir: Directory to save debug information (query_outputs, doc_outputs, loss)
    """
    # Initialize wandb
    wandb.init(
        project="colpali-finetune",
        name="debug-hard-negatives",
        mode="online",
        config={
            "model_name": model_name,
            "data_path": data_path,
            "quantization_strategy": quantization_strategy,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "test_size": test_size,
            "seed": seed,
        }
    )

    # Load model and processor
    print("Loading model and processor...")
    model, processor, device = load_model_and_processor(
        model_name=model_name,
        quantization_strategy=quantization_strategy
    )

    # Load training data
    print("\nLoading training data...")
    train_ds, test_ds = load_training_data(
        data_path=data_path,
        test_size=test_size,
        seed=seed
    )
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    # Create collate function
    collate_fn = create_collate_fn(processor, model)

    # Pre-training inference
    if run_pre_training_inference:
        print("\n" + "="*80)
        print("Running inference BEFORE training...")
        print("="*80)
        pre_results = run_inference(model, processor, train_ds, n_inference_examples)
        print_inference_results(pre_results)

    # Create trainer
    print("\nCreating trainer...")
    trainer = create_trainer(
        model=model,
        train_dataset=train_ds,
        collate_fn=collate_fn,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        debug_dir=debug_dir
    )

    # Start training
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    train_results = trainer.train()
    print("\nTraining completed!")
    print(train_results)

    # Post-training inference
    if run_post_training_inference:
        print("\n" + "="*80)
        print("Running inference AFTER training...")
        print("="*80)
        post_results = run_inference(model, processor, train_ds, n_inference_examples)
        print_inference_results(post_results)

        # Compare before and after
        if run_pre_training_inference:
            print("\n" + "="*80)
            print("Training Impact:")
            print(f"Accuracy before training: {pre_results['accuracy']:.2%}")
            print(f"Accuracy after training: {post_results['accuracy']:.2%}")
            print(f"Improvement: {(post_results['accuracy'] - pre_results['accuracy']):.2%}")
            print("="*80)


def load_config_from_yaml(config_path: str = "configs/finetune_config.yml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing all configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is not valid YAML
        ValueError: If configuration parameters are invalid
    """
    print(f"Loading configuration from: {config_path}")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Flatten the nested structure for easier access
    flattened_config = {
        # Model parameters
        'model_name': config['model']['name'],
        'quantization_strategy': config['model']['quantization_strategy'],

        # Data parameters
        'data_path': config['data']['path'],
        'test_size': config['data']['test_size'],
        'seed': config['data']['seed'],

        # Training parameters
        'output_dir': config['training']['output_dir'],
        'num_train_epochs': config['training']['num_train_epochs'],
        'per_device_train_batch_size': config['training']['per_device_train_batch_size'],
        'per_device_eval_batch_size': config['training']['per_device_eval_batch_size'],
        'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
        'learning_rate': config['training']['learning_rate'],
        'warmup_steps': config['training']['warmup_steps'],
        'logging_steps': config['training']['logging_steps'],
        'save_total_limit': config['training']['save_total_limit'],

        # Inference parameters
        'run_pre_training_inference': config['inference']['run_pre_training'],
        'run_post_training_inference': config['inference']['run_post_training'],
        'n_inference_examples': config['inference']['n_examples'],

        # Debug parameters
        'debug_dir': config['debug']['dir'],
    }

    # Validate configuration
    _validate_config(flattened_config)

    print("Configuration loaded and validated successfully!")
    return flattened_config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If any configuration parameter is invalid
    """
    # Validate quantization_strategy
    valid_quant_strategies = ['4bit', '8bit', None]
    if config['quantization_strategy'] not in valid_quant_strategies:
        raise ValueError(
            f"Invalid quantization_strategy: {config['quantization_strategy']}. "
            f"Must be one of {valid_quant_strategies}"
        )

    # Validate test_size
    if not 0.0 <= config['test_size'] <= 1.0:
        raise ValueError(
            f"Invalid test_size: {config['test_size']}. Must be between 0.0 and 1.0"
        )

    # Validate positive integers
    positive_int_params = [
        'num_train_epochs', 'per_device_train_batch_size',
        'per_device_eval_batch_size', 'gradient_accumulation_steps',
        'warmup_steps', 'logging_steps', 'save_total_limit',
        'n_inference_examples', 'seed'
    ]

    for param in positive_int_params:
        if config[param] <= 0:
            raise ValueError(f"Invalid {param}: {config[param]}. Must be positive")

    # Validate learning_rate
    if config['learning_rate'] <= 0:
        raise ValueError(
            f"Invalid learning_rate: {config['learning_rate']}. Must be positive"
        )


if __name__ == "__main__":
    # Get config path from command line or use default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/finetune_config.yml"

    # Load configuration from YAML
    config = load_config_from_yaml(config_path)

    # Run main function with config parameters
    main(
        model_name=config['model_name'],
        data_path=config['data_path'],
        quantization_strategy=config['quantization_strategy'],
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        logging_steps=config['logging_steps'],
        save_total_limit=config['save_total_limit'],
        test_size=config['test_size'],
        seed=config['seed'],
        run_pre_training_inference=config['run_pre_training_inference'],
        run_post_training_inference=config['run_post_training_inference'],
        n_inference_examples=config['n_inference_examples'],
        debug_dir=config['debug_dir']
    )
