import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from mingpt_lightning.structured_configs import (
    _register_datamodule_configs,
    _register_model_configs,
    _register_trainer_configs,
)

_register_datamodule_configs()
_register_model_configs()
_register_trainer_configs()


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.datamodule)
    cfg.model.vocab_size = datamodule.train_dataset.get_vocab_size()
    cfg.model.block_size = datamodule.train_dataset.get_block_size()

    model = instantiate(cfg.model)

    trainer: Trainer = instantiate(cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
