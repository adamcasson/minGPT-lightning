
# minGPT Lightning

A refactor of [karpathy/minGPT](https://github.com/karpathy/minGPT) to use PyTorch Lightning + Hydra.

### Usage

Here's how you'd instantiate train a GPT-2 (124M param version) for the adder task:

```python
$ python main.py model=gpt2 datamodule=adder
```

### todos

- consider refactoring LightningModule towards [system pattern instead of model pattern](https://pytorch-lightning.readthedocs.io/en/stable/starter/style_guide.html?highlight=system%20pattern#systems-vs-models)
- add back chargpt task from karpathy/minGPT
- update notebooks
- add tests
