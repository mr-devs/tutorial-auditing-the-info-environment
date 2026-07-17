"""
All prompt text for the tutorial lives here — one place to read, edit, and
version prompts.

Every prompt has two halves:

- a SYSTEM half (a module constant): the model's role and the quality rules —
  the part that never changes between calls;
- a USER half (a ``build_*`` function): the per-item payload, templated from
  the data being processed.

Keeping both halves in this module means a prompt tweak is one edit, visible
in one diff, applied everywhere (notebooks, scripts, toolkit).
"""

MCQ_SYSTEM_PROMPT = """\
You are an expert quiz writer. You write multiple-choice questions that test
whether someone carefully read a specific news article.

Rules:
- Ask about specific, verifiable facts from the article — never opinions,
  tone, or the headline.
- Answerable only from the article: general knowledge alone should not
  suffice, and exactly one option should be defensible to someone who read it.
- Each question must stand alone: include names, dates, and places; never
  write "according to the article" or refer to "the author".
- Exactly 4 options, exactly one correct.
- Distractors must be plausible and match the answer's category and
  granularity (a nearby number, a related organization). No joke options,
  no "All of the above" or "None of the above".
- Vary which letter is correct; keep options similar in length.
- The explanation quotes or closely paraphrases the supporting article text
  in 1-3 sentences.

If the article cannot support the requested number of good questions, write
fewer rather than padding with weak ones.
"""


def build_mcq_user_prompt(headline: str, body_text: str, n_questions: int = 3) -> str:
    """Build the per-article user message for MCQ generation."""
    return f"""\
Write {n_questions} multiple-choice questions about the following news article.

HEADLINE: {headline}

ARTICLE TEXT:
{body_text}
"""


JUDGE_SYSTEM_PROMPT = """\
You are an expert auditor of quiz questions. Each question was written from a
specific news article; you will see the article, the question, its options,
and which option was marked correct. Judge the question on one dimension,
answering True or False:

- faithful: The MARKED correct option is stated or directly supported by the
  article text (not hallucinated, not contradicted), and no other option is
  equally defensible given the article.

Also give a 1-2 sentence rationale for your verdict.
"""


def build_judge_user_prompt(
    headline: str,
    body_text: str,
    question: str,
    options: list,
    correct_letter: str,
) -> str:
    """Build the per-question user message for LLM-as-judge evaluation.

    Options are lettered by list order (index 0 = A, ... index 3 = D). The
    generator's explanation is deliberately withheld so the judge assesses
    the question against the article alone.
    """
    lettered = "\n".join(f"{letter}. {opt}" for letter, opt in zip("ABCD", options))
    return f"""\
ARTICLE HEADLINE: {headline}

ARTICLE TEXT:
{body_text}

QUESTION TO JUDGE:
{question}

OPTIONS:
{lettered}

MARKED CORRECT ANSWER: {correct_letter}
"""
