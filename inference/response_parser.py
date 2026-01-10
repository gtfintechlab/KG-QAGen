"""
Response parsing utilities for extracting structured data from LLM outputs.
Handles JSON and list-based response parsing with error handling.
"""

import json
import logging
import re


def parse_llm_json(raw_response: str, num_questions: int) -> dict:
    """
    Parse LLM response expecting JSON format.
    Handles both clean JSON and JSON embedded in markdown code blocks.

    Expected format:
    ```json
    {
      "answers": [
        {"question_index": 1, "answer": "..."},
        ...
      ]
    }
    ```

    Args:
        raw_response: Raw text response from LLM
        num_questions: Expected number of questions

    Returns:
        Dictionary mapping question indices to answers {1: "answer1", 2: "answer2", ...}
        Returns "LLM parse error" for questions that couldn't be parsed
    """
    default_result = {
        i: "LLM parse error" for i in range(1, num_questions + 1)}

    match = re.search(r"```json\s*(.*?)\s*```", raw_response, re.DOTALL)
    if match:
        cleaned_response = match.group(1).strip()
    else:
        cleaned_response = raw_response.strip()

    logging.debug(f"Parsing LLM JSON: {cleaned_response}")

    try:
        data = json.loads(cleaned_response)
        if "answers" not in data:
            logging.warning("No 'answers' key found in the JSON response.")
            return default_result

        answers = data["answers"]
        for ans in answers:
            idx = ans.get("question_index")
            content = ans.get("answer", "")
            if isinstance(idx, int) and 1 <= idx <= num_questions:
                default_result[idx] = content

        return default_result
    except json.JSONDecodeError as e:
        logging.warning(f"JSON parse error: {e}")
        return default_result


def parse_llm_list(raw_response: str, num_questions: int) -> dict:
    """
    Alternative parser for list-formatted responses (fallback for certain models).
    Expects format like: "Q1: answer\nQ2: answer\n..."

    Args:
        raw_response: Raw text response from LLM
        num_questions: Expected number of questions

    Returns:
        Dictionary mapping question indices to answers
    """
    default_result = {
        i: "LLM parse error" for i in range(1, num_questions + 1)}

    question_responses = raw_response.split("\n")
    for i in range(len(question_responses)):
        try:
            question_index = int(question_responses[i].split(':')[0][1:])
            question_response = question_responses[i].split(':')[1][1:]
            default_result[question_index] = question_response
        except:
            logging.warning(f'Issue parsing question {i}')

    return default_result
