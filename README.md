# text2tags

Text2tags is a Python library that uses a finetune of Llama-7b and llama cpp to predict Danbooru tags from natural text.

## Installation
Use the package manager [pip]
```
pip install text2tags-ooferdoodles
```

## Usage

```python
import text2tags

model = text2tags.TaggerLlama()

tags = model.predict_tags("Minato Aqua from hololive with pink and blue twintails in a blue maid outfit")

print(', '.join(tags))
```

## License

[MIT](https://choosealicense.com/licenses/mit/)