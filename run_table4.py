"""
Table 4 재현 실험: CoT-decoding vs 다양한 디코딩 전략 비교

지원 모델/데이터셋:
- GSM8K (openai/gsm8k) + Llama-3.1-8B 등 (자유응답, 논문 원본 설정)
- KMMLU Math (HAERAE-HUB/KMMLU) + kanana-8b 등 (MCQA)

디코딩 전략:
1. Top-k sampling (k=10)
2. Top-p / Nucleus sampling (p=0.9)
3. Beam search (b=10)
4. Temperature sampling (T=0.7)
5. Greedy decoding
6. Self-consistency w/o CoT prompt (10 paths)
7. CoT-decoding (k=10)
8. CoT-decoding aggregation (k=10)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_decoding import (
    MCQAExtractor,
    FreeFormExtractor,
    AnswerExtractor,
    cot_decoding,
    cot_decoding_aggregate,
    greedy_decode,
    self_consistency_decode,
)


def format_mcqa_prompt(question: str, A: str, B: str, C: str, D: str) -> str:
    """KMMLU MCQA 문제를 프롬프트로 변환 (논문의 Q:\\nA: 형식)"""
    return (
        f"Q: {question}\n"
        f"A. {A}\n"
        f"B. {B}\n"
        f"C. {C}\n"
        f"D. {D}\n"
        f"A:"
    )


def format_gsm8k_prompt(question: str) -> str:
    """GSM8K 문제를 프롬프트로 변환 (논문 형식: Q:\\nA:)"""
    return f"Q: {question}\nA:"


def extract_gsm8k_gold(answer_text: str) -> str:
    """GSM8K answer 필드에서 '#### 숫자' 형태의 정답을 추출."""
    import re
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


# ── 디코딩 전략 함수들 ──


def _sampling_loop(model, input_ids, max_new_tokens, eos_token_id, sample_fn,
                   tokenizer=None, extractor: AnswerExtractor | None = None):
    """KV cache 기반 공통 샘플링 루프."""
    generated = []
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values

    next_token = sample_fn(logits)
    generated.append(next_token.item())

    if next_token.item() == eos_token_id:
        return generated
    if extractor and tokenizer and extractor.is_stop_token(next_token.item(), tokenizer):
        return generated

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                next_token.view(1, 1), past_key_values=past_kv, use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values

        next_token = sample_fn(logits)
        generated.append(next_token.item())

        if next_token.item() == eos_token_id:
            break
        if extractor and tokenizer and extractor.is_stop_token(next_token.item(), tokenizer):
            break

    return generated


def decode_greedy(model, tokenizer, input_ids, extractor, max_new_tokens=256) -> str | None:
    tokens, _ = greedy_decode(
        model, input_ids, max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer, extractor=extractor,
    )
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return extractor.extract(text)


def decode_temperature(model, tokenizer, input_ids, extractor, T=0.7, max_new_tokens=256) -> str | None:
    def sample_fn(logits):
        probs = F.softmax(logits / T, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    tokens = _sampling_loop(model, input_ids, max_new_tokens, tokenizer.eos_token_id, sample_fn,
                            tokenizer=tokenizer, extractor=extractor)
    return extractor.extract(tokenizer.decode(tokens, skip_special_tokens=True))


def decode_top_k_sampling(model, tokenizer, input_ids, extractor, k=10, max_new_tokens=256) -> str | None:
    def sample_fn(logits):
        top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)
        probs = F.softmax(top_k_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        return top_k_indices.gather(-1, sampled_idx).squeeze(-1)

    tokens = _sampling_loop(model, input_ids, max_new_tokens, tokenizer.eos_token_id, sample_fn,
                            tokenizer=tokenizer, extractor=extractor)
    return extractor.extract(tokenizer.decode(tokens, skip_special_tokens=True))


def decode_nucleus(model, tokenizer, input_ids, extractor, p=0.9, max_new_tokens=256) -> str | None:
    def sample_fn(logits):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_logits[sorted_indices_to_remove] = float('-inf')
        probs = F.softmax(sorted_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        return sorted_indices.gather(-1, sampled_idx).squeeze(-1)

    tokens = _sampling_loop(model, input_ids, max_new_tokens, tokenizer.eos_token_id, sample_fn,
                            tokenizer=tokenizer, extractor=extractor)
    return extractor.extract(tokenizer.decode(tokens, skip_special_tokens=True))


def decode_beam_search(model, tokenizer, input_ids, extractor, num_beams=10, max_new_tokens=256) -> str | None:
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extractor.extract(text)


def decode_self_consistency(model, tokenizer, input_ids, extractor, n_paths=10, max_new_tokens=256) -> str | None:
    path = self_consistency_decode(
        model, tokenizer, input_ids,
        n_paths=n_paths, max_new_tokens=max_new_tokens, extractor=extractor,
    )
    return path.answer


def decode_cot(model, tokenizer, input_ids, extractor, k=10, max_new_tokens=256) -> str | None:
    path = cot_decoding(
        model, tokenizer, input_ids,
        k=k, max_new_tokens=max_new_tokens, extractor=extractor,
    )
    return path.answer


def decode_cot_aggregate(model, tokenizer, input_ids, extractor, k=10, max_new_tokens=256) -> str | None:
    path = cot_decoding_aggregate(
        model, tokenizer, input_ids,
        k=k, max_new_tokens=max_new_tokens, extractor=extractor,
    )
    return path.answer


# ── 실험 설정 ──

STRATEGIES = {
    "top_k_sampling": {
        "fn": decode_top_k_sampling,
        "label": "Top-k sampling (k=10)",
        "kwargs": {"k": 10},
    },
    "nucleus_sampling": {
        "fn": decode_nucleus,
        "label": "Top-p / Nucleus sampling (p=0.9)",
        "kwargs": {"p": 0.9},
    },
    "beam_search": {
        "fn": decode_beam_search,
        "label": "Beam search (b=10)",
        "kwargs": {"num_beams": 10},
    },
    "temperature_sampling": {
        "fn": decode_temperature,
        "label": "Temperature sampling (T=0.7)",
        "kwargs": {"T": 0.7},
    },
    "greedy": {
        "fn": decode_greedy,
        "label": "Greedy decoding",
        "kwargs": {},
    },
    "self_consistency": {
        "fn": decode_self_consistency,
        "label": "Self-consistency w/o CoT prompt (10 paths)",
        "kwargs": {"n_paths": 10},
    },
    "cot_decoding": {
        "fn": decode_cot,
        "label": "CoT-decoding (k=10)",
        "kwargs": {"k": 10},
    },
    "cot_decoding_agg": {
        "fn": decode_cot_aggregate,
        "label": "CoT-decoding (agg, k=10)",
        "kwargs": {"k": 10},
    },
}

ANSWER_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}


def evaluate_strategy(
    model,
    tokenizer,
    dataset,
    strategy_name: str,
    extractor: AnswerExtractor,
    dataset_type: str = "kmmlu",
    max_new_tokens: int = 64,
    num_samples: int = 200,
    device: str = "cuda",
) -> dict:
    """특정 디코딩 전략으로 데이터셋 평가."""
    strategy = STRATEGIES[strategy_name]
    fn = strategy["fn"]
    kwargs = strategy["kwargs"]
    label = strategy["label"]

    correct = 0
    total = 0
    results = []

    samples = list(dataset)[:num_samples]

    for i, sample in enumerate(tqdm(samples, desc=f"[{label}]")):
        if dataset_type == "gsm8k":
            prompt = format_gsm8k_prompt(sample["question"])
            gold_answer = extract_gsm8k_gold(sample["answer"])
        else:
            prompt = format_mcqa_prompt(
                sample["question"], sample["A"], sample["B"], sample["C"], sample["D"]
            )
            gold_answer = ANSWER_MAP[sample["answer"]]

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        try:
            predicted = fn(model, tokenizer, input_ids, extractor,
                           max_new_tokens=max_new_tokens, **kwargs)
        except Exception as e:
            predicted = None
            print(f"  [Error] sample {i}: {e}")

        is_correct = predicted == gold_answer
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": i,
            "question": sample["question"][:80],
            "gold": gold_answer,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "strategy": strategy_name,
        "label": label,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def print_table4(all_results: list[dict], model_name: str, dataset_name: str):
    print("\n" + "=" * 60)
    print("Table 4 | Decoding Strategy Comparison")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    sorted_results = sorted(all_results, key=lambda x: x["accuracy"])

    for r in sorted_results:
        bar = "█" * int(r["accuracy"] * 40)
        print(f"  {r['label']:<45} {r['accuracy']*100:5.1f}%  {bar}")

    print("=" * 60)


DATASET_CONFIGS = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "test",
        "display_name": "GSM8K",
        "default_max_new_tokens": 256,
    },
    "kmmlu": {
        "hf_path": "HAERAE-HUB/KMMLU",
        "hf_name": "Math",
        "split": "test",
        "display_name": "KMMLU Math",
        "default_max_new_tokens": 64,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Table 4 Reproduction: CoT-decoding comparison")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="데이터셋 선택: gsm8k (자유응답) / kmmlu (MCQA)")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="최대 생성 토큰 수 (미지정 시 데이터셋 기본값 사용)")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES.keys()),
                        choices=list(STRATEGIES.keys()),
                        help="실행할 디코딩 전략 목록")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ds_config = DATASET_CONFIGS[args.dataset]
    max_new_tokens = args.max_new_tokens or ds_config["default_max_new_tokens"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # 데이터셋 로드
    print(f"Loading dataset: {ds_config['hf_path']} ({ds_config['hf_name']})")
    dataset = load_dataset(ds_config["hf_path"], ds_config["hf_name"], split=ds_config["split"])
    print(f"  Total test samples: {len(dataset)}, using: {args.num_samples}")
    print(f"  Max new tokens: {max_new_tokens}")

    # 태스크별 추출기
    if args.dataset == "gsm8k":
        extractor = FreeFormExtractor()
    else:
        extractor = MCQAExtractor(choices=["A", "B", "C", "D"])

    # 실험 실행
    all_results = []
    total_start = time.time()

    for strategy_name in args.strategies:
        print(f"\n{'─' * 50}")
        print(f"Running: {STRATEGIES[strategy_name]['label']}")
        print(f"{'─' * 50}")

        start = time.time()
        result = evaluate_strategy(
            model, tokenizer, dataset,
            strategy_name=strategy_name,
            extractor=extractor,
            dataset_type=args.dataset,
            max_new_tokens=max_new_tokens,
            num_samples=args.num_samples,
            device=args.device,
        )
        elapsed = time.time() - start
        result["elapsed_seconds"] = elapsed

        print(f"  Accuracy: {result['accuracy']*100:.1f}% ({result['correct']}/{result['total']})")
        print(f"  Time: {elapsed:.1f}s")

        all_results.append(result)

        save_path = output_dir / f"{strategy_name}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - total_start

    print_table4(all_results, args.model, ds_config["display_name"])
    print(f"\nTotal time: {total_elapsed:.1f}s")

    summary = {
        "model": args.model,
        "dataset": ds_config["display_name"],
        "num_samples": args.num_samples,
        "max_new_tokens": max_new_tokens,
        "total_elapsed_seconds": total_elapsed,
        "results": [
            {
                "strategy": r["strategy"],
                "label": r["label"],
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"],
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for r in sorted(all_results, key=lambda x: x["accuracy"])
        ],
    }

    summary_path = output_dir / "table4_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
