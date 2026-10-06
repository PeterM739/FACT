import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from .base import ModelAdapter


class TransformersAdapter(ModelAdapter):
    def __init__(self, spec):
        cuda = torch.cuda.is_available() and spec.device != "cpu"
        default = (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16) if cuda else torch.float32
        dtype = getattr(torch, spec.dtype) if spec.dtype else default
        self.tokenizer = AutoTokenizer.from_pretrained(spec.name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        device = "cuda" if cuda else "cpu"
        if spec.quantization: 
            bits = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype) if spec.quantization == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(spec.name, dtype=dtype, quantization_config=bits, device_map={"": device}).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(spec.name, dtype=dtype, device_map="auto" if cuda else {"": device}).eval()
            if cuda and {"cpu", "disk"} & set(getattr(self.model, "hf_device_map", {}).values()):
                raise RuntimeError(f"{spec.name} does not fit on the visible GPUs")
        self.dtype = str(dtype).removeprefix("torch.")
        gpus = torch.cuda.device_count()
        self.environment = {"gpu": (f"{gpus}× " if gpus > 1 else "") + torch.cuda.get_device_name() if cuda else None,
                            "torch": str(torch.__version__), 
                            "cuda": torch.version.cuda}

    def generate(self, batch: list[list[dict]], temperature: float, max_new_tokens: int, seed: int) -> list[str]:
        set_seed(seed)
        texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0} if temperature > 0 else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                                      **sampling)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in out[:, inputs["input_ids"].shape[1]:]]

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
