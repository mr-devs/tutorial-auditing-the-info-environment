"""All prompt text for the tutorial — one place to read, edit, and version
prompts.

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
- Every question must target the central news story — the new development
  the article exists to report. Background mentioned in passing (what a
  government department does, who holds an office, standing facts about a
  person, place, or organization) is off-limits even though the article
  states it: such facts can be known without reading the article.
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


def build_mcq_user_prompt(headline: str, body_text: str, n_questions: int = 1) -> str:
    """Build the per-article user message for MCQ generation.

    Parameters
    ----------
    headline : str
        The article headline.
    body_text : str
        The full article body text.
    n_questions : int, default 1
        Number of questions to request.

    Returns
    -------
    str
        The formatted user message.
    """
    plural = "s" if n_questions != 1 else ""
    return f"""\
Write {n_questions} multiple-choice question{plural} about the following news article.

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

    Parameters
    ----------
    headline : str
        The source article headline.
    body_text : str
        The full source article body text.
    question : str
        The question to judge.
    options : list
        The answer options, lettered by list order (index 0 = A, ...
        index 3 = D).
    correct_letter : str
        The letter marked as the correct answer.

    Returns
    -------
    str
        The formatted user message.

    Notes
    -----
    The generator's explanation is deliberately withheld so the judge
    assesses the question against the article alone.
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


ANSWER_SYSTEM_PROMPT = """\
You are an expert news-quiz contestant. Each question was written from a
recently published news article (within the last few weeks). You are NOT
given the article — answer from what you know or can find.

Rules:
- Pick the single best option: exactly one of A, B, C, or D.
- Always commit to one letter, even if you are unsure.
- Give 1-2 sentences of reasoning and a confidence between 0 and 1.
"""

# Appended to ANSWER_SYSTEM_PROMPT for the web_search method only.
ANSWER_WEBSEARCH_ADDENDUM = """\
- You have a web-search tool. Search to verify before answering, and
  list the URLs of the sources you relied on in `citations`.
"""


def _letter_options(options: list) -> str:
    """Render options as lettered lines (index 0 = A, ... index 3 = D)."""
    return "\n".join(f"{letter}. {opt}" for letter, opt in zip("ABCD", options))


def build_answer_user_prompt(question: str, options: list) -> str:
    """Build the per-question user message for the answering conditions.

    Deliberately contains ONLY the question and its lettered options — no
    article headline or text — so closed-book answers reflect the model's
    own knowledge and web-search answers reflect its retrieval.

    Parameters
    ----------
    question : str
        The question to answer.
    options : list
        The answer options, lettered by list order (index 0 = A, ...
        index 3 = D).

    Returns
    -------
    str
        The formatted user message.
    """
    return f"""\
QUESTION:
{question}

OPTIONS:
{_letter_options(options)}
"""


DEBATE_REVISION_SYSTEM_PROMPT = """\
You are an expert news-quiz contestant. Each question was written from a
recently published news article (within the last few weeks). You are NOT
given the article — answer from what you know.

Other contestants answered the same question; their answers and reasoning
are shown to you as additional advice. Weigh their arguments against your
own knowledge, then give your own (possibly revised) final answer.

Rules:
- Pick the single best option: exactly one of A, B, C, or D.
- Always commit to one letter, even if you are unsure.
- Give 1-2 sentences of reasoning and a confidence between 0 and 1.
"""


def build_debate_revision_prompt(
    question: str,
    options: list,
    peer_answers: list,
) -> str:
    """Build the revision-round user message for the debate condition.

    Parameters
    ----------
    question : str
        The question to answer.
    options : list
        The answer options, lettered by list order (index 0 = A, ...
        index 3 = D).
    peer_answers : list of dict
        The OTHER agents' most recent answers, each with keys ``"agent"``,
        ``"answer_letter"``, ``"confidence"``, and ``"reasoning"``.

    Returns
    -------
    str
        The formatted user message.

    Notes
    -----
    Each agent sees only the other agents' previous-round answers (the
    standard society-of-minds setup), keeping prompt growth linear in the
    number of rounds.
    """
    peers = "\n".join(
        f"Agent {p['agent']} answered {p['answer_letter']} "
        f"(confidence {p['confidence']:.2f}): {p['reasoning']}"
        for p in peer_answers
    )
    return f"""\
QUESTION:
{question}

OPTIONS:
{_letter_options(options)}

PEER ANSWERS:
{peers}
"""
