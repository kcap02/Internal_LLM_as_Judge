"""Prompt construction for all three judging formats.

Formats (see items.py for the unified item schema):
  mcq      — question + choices + proposed letter  -> verdict Yes/No
  single   — question + one free-text response     -> verdict Yes/No
  pairwise — question + two responses A/B          -> verdict A/B

Solver prompts reproduce the canonical Hendrycks MMLU eval.py style (raw
prompt, subject header, "Answer:" suffix, no chat template, single forward
pass). Judge prompts keep the same register so verdicts are read from
next-token logprobs at the position following "Answer:".

Chat-template mode is available (use_chat_template=True) for instruct-native
judging; the readout is unchanged because the rendered string still ends with
"Answer:" as the generation prefix.
"""

from __future__ import annotations

import string

ALL_LETTERS = list(string.ascii_uppercase)  # supports up to 26 options (MMLU-Pro uses 10)


def letters_for(n_choices: int) -> list[str]:
    return ALL_LETTERS[:n_choices]


def format_subject(subject: str) -> str:
    return " " + subject.replace("_", " ")


# ── Solver (MCQ) ──────────────────────────────────────────────────────────────
def format_mcq_example(question: str, choices: list, letter: str | None = None) -> str:
    ls = letters_for(len(choices))
    lines = [question] + [f"{l}. {c}" for l, c in zip(ls, choices)]
    prompt = "\n".join(lines) + "\nAnswer:"
    if letter is not None:
        prompt += f" {letter}\n\n"
    return prompt


def gen_solver_prompt(dev_examples: list, subject: str, k: int) -> str:
    header = ("The following are multiple choice questions (with answers) about"
              f"{format_subject(subject)}.\n\n")
    body = "".join(
        format_mcq_example(ex["question"], ex["choices"],
                           letters_for(len(ex["choices"]))[ex["answer"]])
        for ex in dev_examples[:k]
    )
    return header + body


# ── Judge: MCQ (Yes/No) ───────────────────────────────────────────────────────
def format_mcq_judge_example(question: str, choices: list, proposed_letter: str,
                             verdict: str | None = None) -> str:
    ls = letters_for(len(choices))
    lines = [question] + [f"{l}. {c}" for l, c in zip(ls, choices)]
    prop_text = choices[ls.index(proposed_letter)]
    lines.append(f"Proposed answer: {proposed_letter}. {prop_text}")
    prompt = "\n".join(lines) + "\nIs the proposed answer correct? Answer:"
    if verdict is not None:
        prompt += f" {verdict}\n\n"
    return prompt


def gen_mcq_judge_header(subject: str) -> str:
    return ("The following are multiple choice questions about"
            f"{format_subject(subject)}, each followed by a proposed answer. "
            "Decide whether the proposed answer is correct (Yes or No).\n\n")


# ── Judge: single free-text response (Yes/No) ─────────────────────────────────
SINGLE_JUDGE_HEADER = (
    "You will be shown an instruction and a candidate response. Decide whether "
    "the response is a correct, faithful and high-quality answer to the "
    "instruction (Yes or No).\n\n"
)


def format_single_judge_example(question: str, response: str,
                                verdict: str | None = None) -> str:
    prompt = (f"Instruction:\n{question}\n\n"
              f"Response:\n{response}\n\n"
              "Is the response a correct and high-quality answer? Answer:")
    if verdict is not None:
        prompt += f" {verdict}\n\n"
    return prompt


# ── Judge: pairwise (A/B) ─────────────────────────────────────────────────────
PAIRWISE_JUDGE_HEADER = (
    "You will be shown an instruction and two candidate responses, A and B. "
    "Decide which response answers the instruction better (A or B).\n\n"
)


def format_pairwise_judge_example(question: str, response_a: str, response_b: str,
                                  verdict: str | None = None) -> str:
    prompt = (f"Instruction:\n{question}\n\n"
              f"Response A:\n{response_a}\n\n"
              f"Response B:\n{response_b}\n\n"
              "Which response is better? Answer:")
    if verdict is not None:
        prompt += f" {verdict}\n\n"
    return prompt


# ── Unified item -> (header, body) ────────────────────────────────────────────
def build_judge_prompt(item: dict) -> tuple[str, str]:
    """Return (header, body). The task-token subgraph for spectral analysis
    starts where the body starts — keep the two parts separate so the caller
    can compute the token index boundary."""
    fmt = item["format"]
    if fmt == "mcq":
        header = gen_mcq_judge_header(item.get("subject", ""))
        body = format_mcq_judge_example(item["question"], item["choices"],
                                        item["proposed_letter"])
    elif fmt == "single":
        header = SINGLE_JUDGE_HEADER
        body = format_single_judge_example(item["question"], item["response"])
    elif fmt == "pairwise":
        header = PAIRWISE_JUDGE_HEADER
        body = format_pairwise_judge_example(item["question"],
                                             item["response_a"],
                                             item["response_b"])
    else:
        raise ValueError(f"Unknown item format: {fmt!r}")
    return header, body


def verdict_labels(fmt: str) -> list[str]:
    return ["A", "B"] if fmt == "pairwise" else ["Yes", "No"]


def render(header: str, body: str, tokenizer=None,
           use_chat_template: bool = False) -> str:
    """Assemble the final prompt string.

    Raw mode simply concatenates. Chat mode wraps header+body as a user turn
    and appends the assistant generation prefix; the string still ends with
    "Answer:" so the logprob readout position is identical.
    """
    text = header + body
    if not use_chat_template:
        return text
    if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
        return text
    # The trailing "Answer:" must stay at the very end of the rendered string:
    # move it out of the user turn into the assistant prefix.
    assert text.endswith("Answer:"), "judge prompts must end with 'Answer:'"
    user_part = text[: -len("Answer:")].rstrip()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_part}],
        tokenize=False, add_generation_prompt=True,
    )
    return rendered + "Answer:"


def task_token_start(header: str, body: str, tokenizer, rendered: str) -> int:
    """Index of the first task (body) token in the rendered prompt.

    Used for spectral_trust subgraph_indices: restricting the attention graph
    to task tokens controls for header/preamble length — a key nuisance for
    length-sensitive spectral metrics.

    Works by tokenizing the rendered prefix that precedes the body. Exact for
    raw mode; for chat mode it is exact as long as the template does not
    re-tokenize across the header/body boundary (assert below catches drift).
    """
    pos = rendered.find(body[:64])  # locate body inside the rendered string
    if pos <= 0:
        return 0
    prefix_ids = tokenizer(rendered[:pos], add_special_tokens=True)["input_ids"]
    return len(prefix_ids)
