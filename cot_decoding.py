"""
CoT-Decoding 구현 모듈 (범용)

논문: "Chain-of-Thought Reasoning without Prompting" (Wang & Zhou, 2024)

지원 태스크 유형:
- MCQA (선택지 개수 자유)
- 자유응답 (숫자, 텍스트 등)
- Yes/No

답변 추출은 AnswerExtractor 프로토콜을 통해 플러그인 방식으로 제공.
"""

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ── 답변 추출 프로토콜 ──


class AnswerExtractor(ABC):
    """태스크별 답변 추출 인터페이스."""

    @abstractmethod
    def extract(self, text: str) -> str | None:
        """생성된 텍스트에서 답변을 추출."""
        ...

    @abstractmethod
    def get_stop_token_ids(self, tokenizer: PreTrainedTokenizerBase) -> set[int]:
        """early stopping에 사용할 토큰 ID 집합. 빈 set이면 early stopping 없음."""
        ...

    def is_stop_token(self, token_id: int, tokenizer: PreTrainedTokenizerBase) -> bool:
        """토큰이 stop 토큰인지 확인."""
        stop_ids = self.get_stop_token_ids(tokenizer)
        return token_id in stop_ids if stop_ids else False


class MCQAExtractor(AnswerExtractor):
    """MCQA 답변 추출기. 선택지 개수를 자유롭게 지정 가능."""

    def __init__(self, choices: list[str] | None = None):
        """
        Args:
            choices: 선택지 레이블 리스트. 기본값 ["A", "B", "C", "D"].
        """
        self.choices = choices or ["A", "B", "C", "D"]
        self._stop_ids: set[int] | None = None

    def extract(self, text: str) -> str | None:
        text = text.strip()
        choice_set = set(self.choices)

        # 첫 글자가 선택지인 경우
        if text and text[0] in choice_set:
            return text[0]

        # "답: X", "정답: X", "answer: X" 패턴
        choice_pattern = "|".join(re.escape(c) for c in self.choices)
        patterns = [
            rf'(?:답|정답|answer|Answer)\s*[:：]?\s*({choice_pattern})',
            rf'\(({choice_pattern})\)',
            rf'({choice_pattern})\.',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).upper()

        return None

    def get_stop_token_ids(self, tokenizer: PreTrainedTokenizerBase) -> set[int]:
        if self._stop_ids is None:
            ids = set()
            for choice in self.choices:
                for variant in (choice, f" {choice}"):
                    encoded = tokenizer.encode(variant, add_special_tokens=False)
                    ids.update(encoded)
            self._stop_ids = ids
        return self._stop_ids


class FreeFormExtractor(AnswerExtractor):
    """자유응답 추출기 (GSM8K 스타일 숫자, 일반 텍스트)."""

    def __init__(self, answer_pattern: str | None = None):
        """
        Args:
            answer_pattern: 답변 추출 정규식. 기본값은 숫자 추출.
                첫 번째 캡처 그룹이 답변으로 사용됨.
        """
        self.answer_pattern = answer_pattern or r'####\s*(.+?)(?:\s*$|\n)'
        self._fallback_pattern = r'(-?[\d,]+\.?\d*)'

    def extract(self, text: str) -> str | None:
        # 기본 패턴
        match = re.search(self.answer_pattern, text)
        if match:
            return match.group(1).strip()

        # fallback: 마지막 숫자
        matches = re.findall(self._fallback_pattern, text)
        if matches:
            return matches[-1].replace(",", "")

        return None

    def get_stop_token_ids(self, tokenizer: PreTrainedTokenizerBase) -> set[int]:
        # 자유응답은 early stopping 없음 — 전체 생성 후 추출
        return set()


class YesNoExtractor(AnswerExtractor):
    """Yes/No 답변 추출기."""

    _STOP_IDS: set[int] | None = None

    def extract(self, text: str) -> str | None:
        text = text.strip().lower()
        if text.startswith(("yes", "네", "예", "맞")):
            return "yes"
        if text.startswith(("no", "아니", "틀")):
            return "no"

        for pattern in [r'\b(yes|no)\b', r'(네|예|아니오|아니)']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).lower()
                return "yes" if val in ("yes", "네", "예") else "no"

        return None

    def get_stop_token_ids(self, tokenizer: PreTrainedTokenizerBase) -> set[int]:
        if self._STOP_IDS is None:
            ids = set()
            for word in ("yes", "no", "Yes", "No", " yes", " no", " Yes", " No",
                         "네", "아니", " 네", " 아니"):
                encoded = tokenizer.encode(word, add_special_tokens=False)
                ids.update(encoded)
            YesNoExtractor._STOP_IDS = ids
        return self._STOP_IDS


# ── 기본 추출기 (하위호환) ──

_default_extractor = MCQAExtractor()


def extract_answer_from_text(text: str, extractor: AnswerExtractor | None = None) -> str | None:
    """편의 함수: 텍스트에서 답변 추출."""
    return (extractor or _default_extractor).extract(text)


# ── 데이터 클래스 ──


@dataclass
class DecodingPath:
    """하나의 디코딩 경로 결과."""
    text: str
    answer: str | None
    delta: float  # confidence = prob(top1) - prob(top2) at answer token
    tokens: list[int]


# ── 디코딩 함수들 ──


def greedy_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    eos_token_id: int | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    extractor: AnswerExtractor | None = None,
) -> tuple[list[int], list[torch.Tensor]]:
    """Greedy decoding with KV cache. extractor가 주어지면 stop 토큰에서 중단."""
    generated = []
    all_logits = []

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    all_logits.append(logits)
    next_token = logits.argmax(dim=-1)
    generated.append(next_token.item())

    if eos_token_id is not None and next_token.item() == eos_token_id:
        return generated, all_logits
    if extractor and tokenizer and extractor.is_stop_token(next_token.item(), tokenizer):
        return generated, all_logits

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        all_logits.append(logits)
        next_token = logits.argmax(dim=-1)
        generated.append(next_token.item())

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        if extractor and tokenizer and extractor.is_stop_token(next_token.item(), tokenizer):
            break

    return generated, all_logits


def compute_answer_delta(logits_at_answer: torch.Tensor) -> float:
    """답변 위치의 logits에서 Δ(top1 - top2 확률 차이) 계산."""
    probs = F.softmax(logits_at_answer.squeeze(), dim=-1)
    top2 = torch.topk(probs, k=2)
    return (top2.values[0] - top2.values[1]).item()


def find_answer_token_position(
    generated_tokens: list[int],
    tokenizer: PreTrainedTokenizerBase,
    extractor: AnswerExtractor | None = None,
) -> int | None:
    """생성된 토큰에서 답변 토큰 위치를 찾음."""
    ext = extractor or _default_extractor
    stop_ids = ext.get_stop_token_ids(tokenizer)

    if not stop_ids:
        # stop 토큰이 없으면 (자유응답) 마지막 토큰 위치 반환
        return len(generated_tokens) - 1 if generated_tokens else None

    for i, token_id in enumerate(generated_tokens):
        if token_id in stop_ids:
            return i
    return None


def _build_paths_from_topk(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    k: int,
    max_new_tokens: int,
    extractor: AnswerExtractor | None = None,
) -> list[DecodingPath]:
    """top-k 토큰 각각에서 greedy decoding 수행 후 경로 리스트 반환."""
    ext = extractor or _default_extractor

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        first_logits = outputs.logits[:, -1, :]
        base_past = outputs.past_key_values

    top_k_tokens = torch.topk(first_logits, k=k, dim=-1)
    paths: list[DecodingPath] = []

    for i in range(k):
        token_id = top_k_tokens.indices[0, i].item()
        token_tensor = torch.tensor([[token_id]], device=input_ids.device)

        with torch.no_grad():
            outputs = model(token_tensor, past_key_values=base_past, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values

        generated_tokens = []
        step_logits = [logits]

        next_token = logits.argmax(dim=-1)
        generated_tokens.append(next_token.item())

        should_continue = (
            next_token.item() != tokenizer.eos_token_id
            and not ext.is_stop_token(next_token.item(), tokenizer)
        )

        if should_continue:
            for _ in range(max_new_tokens - 2):
                with torch.no_grad():
                    outputs = model(
                        next_token.unsqueeze(0),
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]
                    past_kv = outputs.past_key_values

                step_logits.append(logits)
                next_token = logits.argmax(dim=-1)
                generated_tokens.append(next_token.item())

                if next_token.item() == tokenizer.eos_token_id:
                    break
                if ext.is_stop_token(next_token.item(), tokenizer):
                    break

        all_tokens = [token_id] + generated_tokens
        full_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
        answer = ext.extract(full_text)

        answer_pos = find_answer_token_position(all_tokens, tokenizer, ext)
        if answer_pos is not None and answer_pos < len(step_logits):
            delta = compute_answer_delta(step_logits[answer_pos])
        elif step_logits:
            delta = compute_answer_delta(step_logits[0])
        else:
            delta = compute_answer_delta(first_logits)

        paths.append(DecodingPath(
            text=full_text, answer=answer, delta=delta, tokens=all_tokens,
        ))

    return paths


def cot_decoding(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    k: int = 10,
    max_new_tokens: int = 256,
    extractor: AnswerExtractor | None = None,
) -> DecodingPath:
    """CoT-Decoding — max Δ path 선택."""
    paths = _build_paths_from_topk(
        model, tokenizer, input_ids, k, max_new_tokens, extractor,
    )
    return max(paths, key=lambda p: p.delta)


def cot_decoding_aggregate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    k: int = 10,
    max_new_tokens: int = 256,
    extractor: AnswerExtractor | None = None,
) -> DecodingPath:
    """CoT-Decoding with weighted aggregation: Δ̃_a = Σ_k Δ_{k,a}."""
    paths = _build_paths_from_topk(
        model, tokenizer, input_ids, k, max_new_tokens, extractor,
    )

    answer_scores: dict[str, float] = {}
    for path in paths:
        if path.answer:
            answer_scores[path.answer] = answer_scores.get(path.answer, 0.0) + path.delta

    if answer_scores:
        best_answer = max(answer_scores, key=answer_scores.get)
        best_path = next(p for p in paths if p.answer == best_answer)
        return best_path

    return max(paths, key=lambda p: p.delta)


def self_consistency_decode(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    n_paths: int = 10,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    extractor: AnswerExtractor | None = None,
) -> DecodingPath:
    """Self-consistency without CoT prompt — majority voting."""
    ext = extractor or _default_extractor
    paths: list[DecodingPath] = []

    for _ in range(n_paths):
        generated = []

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            logits = outputs.logits[:, -1, :] / temperature
            past_kv = outputs.past_key_values

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())

        should_continue = (
            next_token.item() != tokenizer.eos_token_id
            and not ext.is_stop_token(next_token.item(), tokenizer)
        )

        if should_continue:
            for _ in range(max_new_tokens - 1):
                with torch.no_grad():
                    outputs = model(
                        next_token, past_key_values=past_kv, use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :] / temperature
                    past_kv = outputs.past_key_values

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token.item())

                if next_token.item() == tokenizer.eos_token_id:
                    break
                if ext.is_stop_token(next_token.item(), tokenizer):
                    break

        text = tokenizer.decode(generated, skip_special_tokens=True)
        answer = ext.extract(text)
        paths.append(DecodingPath(text=text, answer=answer, delta=0.0, tokens=generated))

    answers = [p.answer for p in paths if p.answer]
    if answers:
        most_common = Counter(answers).most_common(1)[0][0]
        best_path = next(p for p in paths if p.answer == most_common)
        return best_path

    return paths[0]
