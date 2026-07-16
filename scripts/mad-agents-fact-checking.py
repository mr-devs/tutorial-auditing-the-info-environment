"""
Multi-agent debate for fact-checking with web search — openai-agents-python.

Design: two adversarial debaters (Advocate argues the claim is TRUE,
Skeptic argues it is FALSE/misleading) exchange arguments for N rounds.
Both debaters have symmetric access to OpenAI's hosted WebSearchTool and
are instructed to ground arguments in retrieved evidence. A Judge then
reads the full transcript and issues a structured verdict.

The Judge deliberately has NO search access: the verdict should reflect
the evidentiary quality of the debate itself, not the Judge's own
independent retrieval. (If you want a search-enabled Judge, treat it as
a separate experimental condition.)

The debate loop is orchestrated deterministically in Python (rather than
via LLM-driven handoffs) because round structure in a debate is fixed —
we don't want the model deciding who speaks next.

Notes:
- WebSearchTool is a hosted tool and requires OpenAI Responses API models
  (the SDK default for OpenAI models), so no extra setup is needed.
- search_context_size trades cost/latency against retrieval depth; "low"
  is usually fine for short debate turns, bump to "medium"/"high" for
  harder claims.

Install:
    pip install openai-agents
    export OPENAI_API_KEY=sk-...

Run:
    python debate_agents_sdk.py "The MMR vaccine causes autism."
"""

import asyncio
import sys

from pydantic import BaseModel, Field

from agents import Agent, Runner, WebSearchTool

MODEL = "gpt-4o-mini"  # swap for a stronger model in real audits
N_ROUNDS = 2
SEARCH_CONTEXT = "low"  # "low" | "medium" | "high"


# ---------------------------------------------------------------------------
# Structured verdict (Agents SDK supports Pydantic output_type natively)
# ---------------------------------------------------------------------------
class Verdict(BaseModel):
    label: str = Field(
        description="One of: TRUE, MOSTLY TRUE, MIXED, MOSTLY FALSE, FALSE, UNVERIFIABLE"
    )
    confidence: float = Field(ge=0, le=1, description="Judge's confidence in the label")
    rationale: str = Field(description="2-4 sentence justification citing debate points")
    key_evidence: list[str] = Field(description="Decisive arguments from the debate")


# ---------------------------------------------------------------------------
# Agents — both debaters get the SAME search tool (symmetric evidence access)
# ---------------------------------------------------------------------------
def make_search_tool() -> WebSearchTool:
    return WebSearchTool(search_context_size=SEARCH_CONTEXT)


advocate = Agent(
    name="Advocate",
    model=MODEL,
    tools=[make_search_tool()],
    instructions=(
        "You are a debater arguing that the claim under discussion is TRUE. "
        "Before arguing, SEARCH THE WEB for supporting evidence. Ground every "
        "argument in what you actually retrieved and name your sources; do not "
        "invent citations. Directly rebut your opponent's most recent points "
        "when they exist — searching for counter-evidence to their sources is "
        "encouraged. If retrieved evidence genuinely does not support the "
        "claim, concede specific points rather than fabricate support. "
        "Max 150 words of argument per turn."
    ),
)

skeptic = Agent(
    name="Skeptic",
    model=MODEL,
    tools=[make_search_tool()],
    instructions=(
        "You are a debater arguing that the claim under discussion is FALSE or "
        "misleading. Before arguing, SEARCH THE WEB for contradicting evidence: "
        "methodological critiques, missing context, scientific consensus. "
        "Ground every argument in what you actually retrieved and name your "
        "sources; do not invent citations. Directly rebut your opponent's most "
        "recent points when they exist — searching to verify their sources is "
        "encouraged. If retrieved evidence genuinely supports the claim, "
        "concede specific points rather than fabricate objections. "
        "Max 150 words of argument per turn."
    ),
)

judge = Agent(
    name="Judge",
    model=MODEL,
    output_type=Verdict,  # SDK validates the model output against the schema
    # No tools: verdict must rest on the debate transcript alone.
    instructions=(
        "You are an impartial fact-checking judge. You will receive a claim and a "
        "debate transcript between an Advocate (arguing TRUE) and a Skeptic "
        "(arguing FALSE), both of whom had web search access. Weigh the arguments "
        "on evidentiary quality: source credibility, specificity, and whether "
        "claims survived rebuttal. Discount rhetoric and unverifiable or vague "
        "citations. Issue a verdict based only on the transcript."
    ),
)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def debate_fact_check(claim: str, n_rounds: int = N_ROUNDS) -> Verdict:
    transcript: list[str] = []

    def transcript_text() -> str:
        return "\n\n".join(transcript) if transcript else "(no arguments yet)"

    for rnd in range(1, n_rounds + 1):
        for agent in (advocate, skeptic):
            prompt = (
                f"CLAIM: {claim}\n\n"
                f"DEBATE SO FAR:\n{transcript_text()}\n\n"
                f"Round {rnd}: search for evidence, then give your next argument."
            )
            result = await Runner.run(agent, prompt)
            turn = f"[Round {rnd} — {agent.name}]\n{result.final_output}"
            transcript.append(turn)
            print(turn, "\n")

    judge_prompt = (
        f"CLAIM: {claim}\n\nFULL DEBATE TRANSCRIPT:\n{transcript_text()}\n\n"
        "Issue your verdict."
    )
    result = await Runner.run(judge, judge_prompt)
    return result.final_output  # -> Verdict instance


async def main() -> None:
    claim = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "The Great Wall of China is visible from space with the naked eye."
    )
    verdict = await debate_fact_check(claim)
    print("=" * 60)
    print(f"VERDICT:    {verdict.label}  (confidence {verdict.confidence:.2f})")
    print(f"RATIONALE:  {verdict.rationale}")
    print("KEY EVIDENCE:")
    for e in verdict.key_evidence:
        print(f"  - {e}")


if __name__ == "__main__":
    asyncio.run(main())

