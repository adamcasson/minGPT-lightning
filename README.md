
# minGPT Lightning

A refactor of [karpathy/minGPT](https://github.com/karpathy/minGPT) to use PyTorch Lightning + Hydra.

The minGPT-lightning library is three files: [mingpt_lightning/model.py](mingpt_lightning/model.py) contains the actual Transformer model definition, Then there are a number of demos and projects that use the library in the `projects` folder:

- `projects/adder` trains a GPT from scratch to add numbers (inspired by the addition section in the GPT-3 paper)
- `projects/chargpt` trains a GPT to be a character-level language model on some input text file
- `demo.ipynb` shows a minimal usage of the `GPT` and `Trainer` in a notebook format on a simple sorting example
- `generate.ipynb` shows how one can load a pretrained GPT2 and generate text given some prompt

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

### References

Code:

- [openai/gpt-2](https://github.com/openai/gpt-2) has the model definition in TensorFlow, but not the training code
- [openai/image-gpt](https://github.com/openai/image-gpt) has some more modern gpt-3 like modification in its code, good reference as well
- [huggingface/transformers](https://github.com/huggingface/transformers) has a [language-modeling example](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling). It is full-featured but as a result also somewhat challenging to trace. E.g. some large functions have as much as 90% unused code behind various branching statements that is unused in the default setting of simple language modeling
- [Lightning-AI/lightning](https://github.com/Lightning-AI/lightning) source code for PyTorch Lightning
- [facebookresearch/hydra](https://github.com/facebookresearch/hydra) source code for Hydra

### License

MIT
