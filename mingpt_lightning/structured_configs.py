from dataclasses import dataclass
from typing import Literal, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# Datamodule configs
@dataclass
class AdderDatasetConfig:
    _target_: str = 'projects.adder.adder.AdditionDataset'
    ndigit: int = 2
    split: str = MISSING


@dataclass
class AdderDataModuleConfig:
    _target_: str = 'mingpt_lightning.datamodule.BasicDataModule'
    train_dataset: AdderDatasetConfig = AdderDatasetConfig(split='train')
    val_dataset: AdderDatasetConfig = AdderDatasetConfig(split='test')
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False


# Model configs
@dataclass
class GPTBaseConfig:
    n_layer: int = MISSING
    n_head: int = MISSING
    n_embd: int = MISSING
    vocab_size: int = MISSING
    block_size: int = MISSING
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    learning_rate: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1


@dataclass
class GPTConfig(GPTBaseConfig):
    _target_: str = 'mingpt_lightning.model.GPT'


@dataclass
class GPTPretrainedConfig(GPTBaseConfig):
    _target_: str = 'mingpt_lightning.model.GPT.from_pretrained'


@dataclass
class OpenAIGPTConfig(GPTConfig):
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


@dataclass
class GPT2Config(GPTConfig):
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


@dataclass
class GPT2PretrainedConfig(GPTPretrainedConfig):
    model_type: str = 'gpt2'
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024


@dataclass
class GPT2MediumConfig(GPTConfig):
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024


@dataclass
class GPT2MediumPretrainedConfig(GPTPretrainedConfig):
    model_type: str = 'gpt2-medium'
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    vocab_size: int = 50257
    block_size: int = 1024


@dataclass
class GPT2LargeConfig(GPTConfig):
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 1280


@dataclass
class GPT2LargePretrainedConfig(GPTPretrainedConfig):
    model_type: str = 'gpt2-large'
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 1280
    vocab_size: int = 50257
    block_size: int = 1024


@dataclass
class GPT2XLConfig(GPTConfig):
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600


@dataclass
class GPT2XLPretrainedConfig(GPTPretrainedConfig):
    model_type: str = 'gpt2-xl'
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600
    vocab_size: int = 50257
    block_size: int = 1024


@dataclass
class Gopher44mConfig(GPTConfig):
    n_layer: int = 8
    n_head: int = 16
    n_embd: int = 512


@dataclass
class GPTMiniConfig(GPTConfig):
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192


@dataclass
class GPTMicroConfig(GPTConfig):
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


@dataclass
class GPTNanoConfig(GPTConfig):
    n_layer: int = 3
    n_head: int = 3
    n_embd: int = 48


@dataclass
class GPTAdderConfig(GPTConfig):
    """extra small model not included in karpathy/minGPT that I find trains quickly and solves the adder task"""

    n_layer: int = 1
    n_head: int = 8
    n_embd: int = 256
    vocab_size: int = 10
    block_size: int = 6


# Trainer configs
@dataclass
class TrainerConfig:
    _target_: str = 'pytorch_lightning.Trainer'
    enable_checkpointing: bool = False


@dataclass
class TrainerCPUConfig(TrainerConfig):
    accelerator: str = 'cpu'


@dataclass
class TrainerGPUConfig(TrainerConfig):
    accelerator: str = 'gpu'
    devices: int = MISSING


def _register_datamodule_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group='datamodule', name='adder', node=AdderDataModuleConfig)


def _register_model_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group='model', name='openai-gpt', node=OpenAIGPTConfig)
    cs.store(group='model', name='gpt2', node=GPT2Config)
    cs.store(group='model', name='gpt2-pretrained', node=GPT2PretrainedConfig)
    cs.store(group='model', name='gpt2-medium', node=GPT2MediumConfig)
    cs.store(group='model', name='gpt2-medium-pretrained', node=GPT2MediumPretrainedConfig)
    cs.store(group='model', name='gpt2-large', node=GPT2LargeConfig)
    cs.store(group='model', name='gpt2-large-pretrained', node=GPT2LargePretrainedConfig)
    cs.store(group='model', name='gpt2-xl', node=GPT2XLConfig)
    cs.store(group='model', name='gpt2-xl-pretrained', node=GPT2XLPretrainedConfig)
    cs.store(group='model', name='gopher-44m', node=Gopher44mConfig)
    cs.store(group='model', name='gpt-mini', node=GPTMiniConfig)
    cs.store(group='model', name='gpt-micro', node=GPTMicroConfig)
    cs.store(group='model', name='gpt-nano', node=GPTNanoConfig)
    cs.store(group='model', name='gpt-adder', node=GPTAdderConfig)


def _register_trainer_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group='trainer', name='cpu', node=TrainerCPUConfig)
    cs.store(group='trainer', name='gpu', node=TrainerGPUConfig)
