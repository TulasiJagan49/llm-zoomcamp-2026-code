import json
import time

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

from db_save import save_conversation
from evaluation_utils import llm_structured_retry
from metrics import LLMCallRecord, calculate_cost


class RelevanceVerdict(BaseModel):
    relevance: Literal["NOT_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str


judge_instructions = """You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question
""".strip()

judge_prompt = """

Question: {question}

Generated Answer: {answer}
""".strip()


def evaluate_relevance(
    question, answer, client=None, model="gpt-5.4-mini", course="llm-zoomcamp"
):
    if client is None:
        client = OpenAI()

    prompt = judge_prompt.format(question=question, answer=answer)
    start_time = time.time()
    result, usage = llm_structured_retry(
        client=client,
        instructions=judge_instructions,
        user_prompt=prompt,
        output_type=RelevanceVerdict,
        model=model,
    )
    end_time = time.time()
    response_time = start_time - end_time

    llm_answer = f"""
    The relevancy of answer is: {result.relevance}.
    And, the explanation of LLM for that relevancy: {result.explanation}
    """

    cost = calculate_cost(usage=usage, model=model)

    save_conversation(
        record=LLMCallRecord(
            model=model,
            prompt=prompt,
            instructions=judge_instructions,
            answer=llm_answer,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cost=cost,
            response_time=response_time,
        ),
        question=question,
        course=course,
    )

    return result.relevance, result.explanation


if __name__ == "__main__":
    load_dotenv()

    question = "Can I still join the course?"
    answer = "Yes, you can still join. The course is self-paced."

    relevance, explanation = evaluate_relevance(question, answer)
    print(relevance)
    print(explanation)
