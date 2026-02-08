import os
import json
import time
import re
import torch
from tqdm import tqdm
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SEED_FILE = "seed_dataset.jsonl"
TRAIN_FILE = "train_dpo_honest.jsonl"
META_FILE = "train_meta_honest.jsonl"
MODEL_DIR = "./clova_model"


GEMINI_MODEL = "gemini-2.5-flash" 
AUG_PER_SEED = 10  
MAX_NEW_TOKENS = 128
MAX_OUTPUT_TOKENS = 160
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client = genai.Client(api_key=GEMINI_API_KEY)



#0.5b모델 로드
print("0.5b모델 로드")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, 
    local_files_only=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map = "auto" if DEVICE == "cuda" else None,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).eval()

#질문 증강
def augment_prompts(seed_prompt, category, k):
    prompt_msg=f"""LLM의 모델의 정보 부족{category} 능력을 테스트하기 위해 {k}개의 질문을 생성하라.
    [원본 시드] {seed_prompt}
[지침]
1. 반드시 질문 내에 '핵심정보(숫자, 시간, 조건 등)' 이 하나 이상 빠져있어야 한다.
2. 질문을 받은 LLM 모델이 '추가 정보 없이 답할 수 없음'을 깨닫도록 교묘하게 구성한다.
3. 단어 하나(예: kakfa)가 아닌 완전한 문장으로 생성한다.
4. JSON 배열 [{{"prompt": "..."}}] 형식으로만 출력한다."""
    
    try:
        res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_msg)
        return res
    except Exception as e:
        print("증강 에러", {e})
        return []
    
if __name__ == "__main__":
    test_seed = "민수는 사과를 좋아합니다. 민수가 오늘 사과를 몇 개 먹었을까요?"
    test_category="Underspecified"
    print("테스트시작")
    result = augment_prompts(test_seed, test_category, 3)

    if result:
        print("\n제미나이 응답 원본")
        print(result.text)
    else:
        print("실패")