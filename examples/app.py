import gradio as gr
from text2tags import TaggerLlama

model = TaggerLlama()


def predict(caption, max_tokens=128, temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1):
    tags = model.predict_tags(caption, max_tokens=max_tokens, temperature=temperature,
                              top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty)
    return ', '.join(tags)

description = """
### Enter a caption to extract danbooru tags from it.
[ ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) ](https://github.com/DatboiiPuntai/text2tags-lib)
"""

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Caption"),
        gr.Slider(0, 256, step=16, value=128, label='max_tokens'),
        gr.Slider(0, 2, step=0.1, value=0.8, label='temperature'),
        gr.Slider(0, 100, step=5, value=40, label='top_k'),
        gr.Slider(0, 2, step=0.05, value=0.95, label='top_p'),
        gr.Slider(0, 5, step=0.1, value=1.1, label='repeat_penalty'),
    ],
    outputs="text",
    title="Text2Tags",
    description=description,
    examples=[
        ["Minato Aqua from hololive with pink and blue twintails in a blue maid outfit"],
    ],
    allow_flagging="never"
)

demo.launch()
