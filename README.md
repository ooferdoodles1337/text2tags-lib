# text2tags

Text2tags is a Python library that uses a finetune of Llama-7b and llama cpp to predict Danbooru tags from natural text.

## Installation
Use the package manager [pip]
```
pip install text2tags-lib
```

## Usage

```python
from text2tags import TaggerLlama

model = TaggerLlama()

tags = model.predict_tags("Minato Aqua from hololive with pink and blue twintails in a blue maid outfit")

print(', '.join(tags))
```

## Demos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DatboiiPuntai/text2tags-lib/blob/master/examples/text2tags_colab.ipynb)
[![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ooferdoodles/text2tags-demo)

## Other links
[Training Code and Data Analysis](https://github.com/DatboiiPuntai/Text2Tags)
[Blog](https://medium.com/p/9b820478a7d8/edit)

## License

[MIT](https://choosealicense.com/licenses/mit/)
