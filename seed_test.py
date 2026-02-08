import torch 
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("./clova_model").resolve()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("hyperseed0.5 loading")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, 
    local_files_only=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map = "auto",
    dtype=torch.bfloat16,
    local_files_only=True
)

model.eval()

def generate(messages, max_new_tokens=256, temperature=0.5):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    #내 질문+모델의답변
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(
        outputs[0][prompt_len:], skip_special_tokens=True
    ).strip()


messages = [
    {"role": "tool_list", "content": ""},
    {
        "role": "system",
        "content": (
            "- 너는 네이버가 만든 AI 언어모델이며 이름은 'CLOVA X SEED 0.5B모델'이다.\n"
            "- 한국어로 간결하고 정확하게 답하라."
        ),
    },
    {"role": "user", "content": "너는 어떤 데이터셋을 위주로 학습된 모델이야? 데이터셋 이름을 알 수 있어?"},
]
print("----------------------")
print(generate(messages))
print("----------------------")