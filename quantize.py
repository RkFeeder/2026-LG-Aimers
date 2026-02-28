import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

baseline_path = r"C:\College\LG Aimers\exaone_baseline"
output_path = r"C:\College\LG Aimers\exaone_optimized"

model = AutoModelForCausalLM.from_pretrained(
    baseline_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = model.half()

model.save_pretrained(output_path, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(
    baseline_path,
    trust_remote_code=True
)
tokenizer.save_pretrained(output_path)