from typing import Optional
from llama_cpp.llama_types import Optional
from typing import List
import wget
import os
import editdistance
import re
from llama_cpp import Llama


def download_model(
    model_url="https://huggingface.co/ooferdoodles/tagger-ggml-7b/resolve/main/ggml-model-q4_2.bin",
    save_path=os.path.join("models", "ggml-model-q4_2.bin"),
):
    os.makedirs("models", exist_ok=True)
    wget.download(model_url, out=save_path)


def load_tags():
    lookups_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "lookups"
    )
    tags_file = os.path.join(lookups_dir, "tags.txt")
    try:
        with open(tags_file, "r") as f:
            tag_dict = [line.strip() for line in f]
        return tag_dict
    except IOError as e:
        print(f"Error loading tag dictionary: {e}")
        return []


def preprocess_tag(tag):
    tag = tag.lower()
    match = re.match(r"^([^()]*\([^()]*\))\s*.*$", tag)
    return match.group(1) if match else tag


def find_closest_tag(tag, threshold, tag_list, cache={}):
    if tag in cache:
        return cache[tag]

    closest_tag = min(tag_list, key=lambda x: editdistance.eval(tag, x))
    if editdistance.eval(tag, closest_tag) <= threshold:
        cache[tag] = closest_tag
        return closest_tag
    else:
        return None


def correct_tags(tags, tag_list, preprocess=False):
    if preprocess:
        tags = (preprocess_tag(x) for x in tags)
    corrected_tags = set()
    for tag in tags:
        threshold = max(1, len(tag) - 10)
        closest_tag = find_closest_tag(tag, threshold, tag_list)
        if closest_tag:
            corrected_tags.add(closest_tag)
    return sorted(list(corrected_tags))


class TaggerLlama(Llama):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_parts: int = -1,
        n_gpu_layers: int = 0,
        seed: int = 1337,
        f16_kv: bool = True,
        logits_all: bool = False,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        embedding: bool = False,
        n_threads: int | None = None,
        n_batch: int = 512,
        last_n_tokens_size: int = 64,
        lora_base: str | None = None,
        lora_path: str | None = None,
        verbose: bool = True,
        tag_list = load_tags()
    ):
        super().__init__(
            model_path,
            n_ctx,
            n_parts,
            n_gpu_layers,
            seed,
            f16_kv,
            logits_all,
            vocab_only,
            use_mmap,
            use_mlock,
            embedding,
            n_threads,
            n_batch,
            last_n_tokens_size,
            lora_base,
            lora_path,
            verbose,
        )
        self.tag_list = tag_list

    def predict_tags(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[List[str]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ):
        prompt = f"### Caption: {prompt}\n### Tags: "

        output = self.create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
        )
        raw_preds = output["choices"][0]["text"]
        pred_tags = [x.strip() for x in raw_preds.split("### Tags:")[-1].split(",")]
        corrected_tags = correct_tags(pred_tags, self.tag_list)
        return corrected_tags
