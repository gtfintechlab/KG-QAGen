"""
Prompt builders for different testing regimes and question processing modes.
Contains functions to build prompts for RAG, GOLD, single question, batch, and combine operations.
"""


def build_RAG_prompt(document_text: str, question: str) -> list[dict]:
    """
    Build prompt for RAG (Retrieval-Augmented Generation) testing regime.

    Args:
        document_text: Retrieved chunks from vector store
        question: Question to answer

    Returns:
        List of message dictionaries for LLM
    """
    user_instructions = (
        "[SYSTEM INPUT]\n"
        "You are a financial expert, and your task is to answer "
        "the question given to you based on the chunks of a credit agreement provided to you. "
        "If you believe the answer is not present among the chunks, say 'Not found'.\n\n"

        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with the answer to your question, nothing else. See the example below.\n\n"

        "The given document:\n"
        "Apple Inc. is a technology company headquartered in Cupertino, California. "
        "It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.\n\n"

        "The given question:\n"
        "Where is the headquarters of Apple Inc.?\n\n"

        "The expected output:\n"
        "Cupertino, California\n\n"

        "[USER INPUT]\n"
        f"Merged chunks:\n{document_text}\n\n"

        "[QUESTION]\n"
        f"{question}\n"
    )

    return [{"role": "user", "content": user_instructions}]


def build_GOLD_prompt(document_text: str, question: str) -> list[dict]:
    """
    Build prompt for GOLD (oracle baseline) testing regime.

    Args:
        document_text: Document pieces guaranteed to contain the answer
        question: Question to answer

    Returns:
        List of message dictionaries for LLM
    """
    user_instructions = (
        "[SYSTEM INPUT]\n"
        "You are a financial expert, and your task is to answer "
        "the question given to you based on the pieces of a credit agreement which are guaranteed to contain the answer. "
        "If you still believe the answer is not present among the chunks, say 'Not found'.\n\n"

        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with the answer to your question, nothing else. See the example below.\n\n"

        "The given document:\n"
        "Apple Inc. is a technology company headquartered in Cupertino, California. "
        "It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.\n\n"

        "The given question:\n"
        "Where is the headquarters of Apple Inc.?\n\n"

        "The expected output:\n"
        "Cupertino, California\n\n"

        "[USER INPUT]\n"
        f"Merged chunks:\n{document_text}\n\n"

        "[QUESTION]\n"
        f"{question}\n"
    )

    return [{"role": "user", "content": user_instructions}]


def build_prompt_single(document_text: str, question: str, question_index: int) -> list[dict]:
    """
    Build prompt for a single question in JSON response format.

    Args:
        document_text: Full document or chunk text
        question: Single question to answer
        question_index: Index of the question (typically 1 for single questions)

    Returns:
        List of message dictionaries for LLM
    """
    user_instructions = (
        "[SYSTEM INPUT]\n"
        "You are a financial expert, and your task is to answer "
        "the question given to you about the provided credit agreement. "
        "If you believe the answer is not present in the agreement, say 'Not found'.\n\n"

        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with valid JSON, nothing else. See the example below.\n\n"

        "The given document:\n"
        "Apple Inc. is a technology company headquartered in Cupertino, California. "
        "It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.\n\n"

        "The given question:\n"
        "Q1: Where is the headquarters of Apple Inc.?\n\n"

        "The expected output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "Cupertino, California"}\n'
        "  ]\n"
        "}\n\n"

        "[USER INPUT]\n"
        f"Document:\n{document_text}\n\n"

        "[QUESTION]\n"
        f"Q1: {question}\n"
    )

    return [{"role": "user", "content": user_instructions}]


def build_prompt_batch(document_text: str, questions: list[str]) -> list[dict]:
    """
    Build prompt for multiple questions in batch mode with JSON response.

    Args:
        document_text: Full document or chunk text
        questions: List of questions to answer

    Returns:
        List of message dictionaries for LLM
    """
    prompt_lines = [
        "[SYSTEM INPUT]\n"
        "You are a financial expert, and your task is to answer "
        "the questions given to you in batches about the provided credit agreement. "
        "If you believe the answer is not present in the agreement, say 'Not found'.\n\n"

        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with valid JSON, nothing else. See the example below.\n\n"

        "The given document:\n"
        "Tesla, Inc. is an American electric vehicle and clean energy company founded in 2003 by Martin Eberhard and Marc Tarpenning. "
        "Elon Musk became the largest investor and later CEO.\n\n"

        "The given questions:\n"
        "Q1: Who founded Tesla?\n"
        "Q2: What year was Tesla founded?\n\n"

        "The expected output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "Martin Eberhard, Marc Tarpenning"},\n'
        '    {"question_index": 2, "answer": "2003"}\n'
        "  ]\n"
        "}\n\n"

        "[USER INPUT]\n"
        f"Document:\n{document_text}\n\n"

        "[QUESTIONS]\n"
    ]

    for i, question in enumerate(questions, start=1):
        prompt_lines.append(f"Q{i}: {question}")

    combined_prompt = "\n".join(prompt_lines)

    return [{"role": "user", "content": combined_prompt}]


def build_prompt_combine_answers(partial_answers: list[str], questions: list[str]) -> list[dict]:
    """
    Build prompt to merge/combine partial answers from multiple chunks into final answers.

    Args:
        partial_answers: List of JSON strings containing partial answers from different chunks
        questions: List of original questions

    Returns:
        List of message dictionaries for LLM with combining instructions
    """
    prompt_lines = [
        "[SYSTEM INPUT]\n"
        "You are a financial expert, and your task is to combine or merge "
        "the provided partial answers, coming from different chunks of a credit agreement, "
        "into a single final answer for each of the questions given to you. "
        "If you believe the answer is not present in the agreement, say 'Not found'.\n\n"

        "[EXPECTED OUTPUT]\n"

        "The given document:\n"
        "Tesla, Inc. is an American electric vehicle and clean energy company founded in 2003 by Martin Eberhard and Marc Tarpenning. "
        "Elon Musk became the largest investor and later CEO.\n\n"

        "The given questions:\n"
        "Q1: Who founded Tesla?\n"
        "Q2: What year was Tesla founded?\n\n"

        "The expected output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "Martin Eberhard, Marc Tarpenning"},\n'
        '    {"question_index": 2, "answer": "2003"}\n'
        "  ]\n"
        "}\n\n"

        "[USER INPUT]\n"
    ]

    for i, ans in enumerate(partial_answers, start=1):
        prompt_lines.append(f"Chunk {i} partial answer JSON:\n{ans}\n")

    prompt_lines.append("\n[QUESTIONS]\n")
    for i, q in enumerate(questions, start=1):
        prompt_lines.append(f"Q{i}: {q}")

    combined_prompt = "\n".join(prompt_lines)

    return [{"role": "user", "content": combined_prompt}]


def build_RAG_planner_prompt(question: str, n_queries: int) -> list[dict]:
    user_instructions = (
        "[SYSTEM INPUT]\n"
        "You are searching within a single credit agreement.\n"
        "Decompose the question into short keyword-style retrieval queries.\n\n"
        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with valid JSON, nothing else.\n\n"
        "{\n"
        '  "subqueries": ["...", "..."]\n'
        "}\n\n"
        f"[RULES]\n"
        f"- Provide at most {n_queries} subqueries.\n"
        "- Subqueries should be short phrases (keywords), not full questions.\n"
        "- Use exact entity strings from the question when possible.\n\n"
        "[QUESTION]\n"
        f"{question}\n"
    )
    return [{"role": "user", "content": user_instructions}]
