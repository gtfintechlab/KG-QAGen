"""
LLM handler module for managing LLM calls with retry logic and error handling.
Provides high-level functions for calling LLMs and parsing their responses.
"""

import logging
import time

from response_parser import parse_llm_json, parse_llm_list


def call_llm_with_retries(llm, messages: list[dict], num_retries: int,
                          wait_time_duration: float, wait_time_enabled: bool,
                          extra_log_info: str = "") -> str:
    """
    Call the LLM up to num_retries times if blank is returned.
    Returns the raw string from LLM (which should be JSON).

    Args:
        llm: Language model instance with invoke() method
        messages: List of message dictionaries to send to LLM
        num_retries: Number of retry attempts
        wait_time_duration: Time to wait between retries (seconds)
        wait_time_enabled: Whether to enforce waiting between calls
        extra_log_info: Additional context for logging (e.g., doc/chunk info)

    Returns:
        Raw output string from LLM, or empty string if all attempts fail
    """
    for attempt in range(num_retries):
        raw_output = None
        try:
            prompt_str = messages[0]['content'] if messages else ""
            logging.info(
                f"LLM call attempt {attempt+1}/{num_retries} {extra_log_info} "
                f"(prompt length: {len(prompt_str)} chars)"
            )

            response = llm.invoke(messages)
            raw_output = response.content.strip() if hasattr(
                response, "content") else str(response).strip()
            print(raw_output)

            if raw_output:
                if wait_time_enabled:
                    time.sleep(wait_time_duration)
                return raw_output
            else:
                logging.warning(
                    f"Got an empty response from LLM. Retrying in 1s...")
                time.sleep(wait_time_duration)
        except Exception as e:
            if raw_output:
                print(raw_output)
            logging.error(f"LLM call error on attempt {attempt+1}: {e}")
            time.sleep(wait_time_duration)

    logging.error("All attempts returned empty response. Giving up.")
    return ""


def get_llm_json_response(llm, messages: list[dict], num_questions: int,
                          num_retries: int, wait_time_duration: float,
                          model_name: str, extra_log_info: str) -> dict:
    """
    Attempts to get a valid JSON parse from the LLM.
    Retries multiple times if the JSON parse fails.

    Args:
        llm: Language model instance
        messages: List of message dictionaries
        num_questions: Expected number of questions in response
        num_retries: Number of retry attempts for parsing
        wait_time_duration: Time to wait between retries (seconds)
        model_name: Name of the model (for special parsing logic)
        extra_log_info: Additional context for logging

    Returns:
        Dictionary mapping question indices to answers
        Returns "LLM parse error" for questions that couldn't be parsed
    """
    default_result = {
        i: "LLM parse error" for i in range(1, num_questions + 1)}
    parsed_result = default_result.copy()

    for parse_attempt in range(num_retries):
        raw_output = call_llm_with_retries(
            llm, messages, num_retries, wait_time_duration, True,
            extra_log_info=extra_log_info)
        if not raw_output:
            logging.warning(
                f"Empty output from LLM (parse attempt {parse_attempt+1}). Retrying...")
            time.sleep(wait_time_duration)
            continue

        parsed_result = parse_llm_json(raw_output, num_questions)
        if any(ans != "LLM parse error" for ans in parsed_result.values()):
            return parsed_result
        else:
            if model_name != 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo':
                logging.warning(
                    f"Parse error (parse attempt {parse_attempt+1}). Retrying LLM call...")
                continue
            logging.warning(
                f"JSON parse error (parse attempt {parse_attempt+1}). Attempting alternative parsing...")
            parsed_result = parse_llm_list(raw_output, num_questions)
            if any(ans != "LLM parse error" for ans in parsed_result.values()):
                return parsed_result
            logging.warning(
                f"Parse error (parse attempt {parse_attempt+1}). Retrying LLM call...")

    logging.error(
        f"All parse attempts failed. Returning default error responses for {num_questions} questions.")
    return default_result
