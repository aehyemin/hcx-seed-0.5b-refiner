import os
import json
import time
import re
import torch
import random
from tqdm import tqdm
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SEED_FILE = "datasets/meta_dataset/seed_dataset.jsonl"
TRAIN_FILE = "datasets/meta_dataset/train_dpo.jsonl"
META_FILE = "datasets/meta_dataset/train_meta.jsonl"
MODEL_DIR = "./clova_model"
SAVE_TRAIN = "datasets/train_dataset"

GEMINI_MODEL = "gemini-2.5-flash" 
AUG_PER_SEED = 40
MAX_NEW_TOKENS = 128
MAX_OUTPUT_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client = genai.Client(api_key=GEMINI_API_KEY)

def extract_json(text):
    if not text:
        return []
    m = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    
def split_data(input_file, train_file, test_file, ratio=0.9):
    #학습용/테스트용 나누기
    if not os.path.exists(input_file):
        print(f"{input_file} 파일이 없음")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    random.shuffle(lines) 
    
    split_idx = int(len(lines) * ratio)
    train_data = lines[:split_idx]
    test_data = lines[split_idx:]
    
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_data)
    with open(test_file, "w", encoding="utf-8") as f:
        f.writelines(test_data)
        
    print(f"분리 완료: 학습용 {len(train_data)}개 / 평가용 {len(test_data)}개")

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
4. JSON 배열 [{{"prompt": "..."}}] 형식으로만 출력한다.
5. 전체 답변의 길이는 4문장 이내로 핵심만 간결하게 대답한다."""
    
    try:
        res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_msg)
        data = extract_json(res.text)
        return data
    except Exception as e:
        print("증강 에러", {e})
        return []
    

#질문에 대한 chosen 답안 생성
def generate_teacher_answer(prompt, rubric):
    msg = f"""너는 정식한 AI 선생님이다. [루브릭]을 100% 준수하여 답변하라.
[루브릭]: {rubric}
[질문]: {prompt}
정답 작성 시, 정보가 부족하면 절대 어림짐작이나 추측하지 말고 무엇이 부족한지 명확하게 짚어라."""
    try: 
        res = client.models.generate_content(model=GEMINI_MODEL, contents=msg)
        return res.text.strip()
    except Exception as e:
        print("선생님 답변 에러", {e})
        return ""

#질문에 대한 rejected 답안 생성
def generate_student_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.8,
            repetition_penalty=1.5,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        seeds = [json.loads(line) for line in f]
    with open(TRAIN_FILE, "w", encoding="utf-8") as f_train, \
         open(META_FILE, "w", encoding="utf-8") as f_meta:
        for seed in tqdm(seeds, desc="전체 시드 처리중"):
            aug_res = augment_prompts(seed["prompt"], "Underspecified", AUG_PER_SEED)
            for item in aug_res:
                prompt = item.get("prompt")
                if not prompt: continue

                chosen = generate_teacher_answer(prompt, seed["rubric"])
                rejected = generate_student_answer(prompt)


                if len(chosen) < 10 or len(rejected) < 10:
                    continue


                f_train.write(json.dumps({
                    "prompt": prompt, "chosen": chosen, "rejected": rejected
                }, ensure_ascii=False) + "\n")
                

                f_meta.write(json.dumps({
                    "category": seed.get("category", "Underspecified"), 
                    "prompt": prompt,
                    "chosen_type": "Teacher(Gemini)", 
                    "rejected_type": "Student(0.5B)"
                }, ensure_ascii=False) + "\n")

                f_train.flush()
                f_meta.flush()
                time.sleep(1.0) 

    print(f"데이터 생성 완료. 생성된 파일: {TRAIN_FILE}")
    print("\n데이터 생성 완료. 분리 작업")
    train_path = os.path.join(SAVE_TRAIN, "train_9.jsonl")
    test_path = os.path.join(SAVE_TRAIN, "test_1.jsonl")
    split_data(TRAIN_FILE, train_path, test_path)