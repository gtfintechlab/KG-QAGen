"""
Question processor module for handling different question processing strategies.
Manages single-question, batch, RAG, and GOLD processing modes with multi-chunk support.
"""

import config
import json
import re

from document_processor import retrieve_vector_db_docs, merge_and_dedupe_docs, load_vector_db_text
from llm_handler import get_llm_json_response
from prompts import (build_RAG_prompt, build_GOLD_prompt, build_prompt_single,
                     build_prompt_batch, build_prompt_combine_answers, build_RAG_planner_prompt)


def process_rag_questions(llm, df, unanswered_indices, doc_id, vector_store,
                          rag_top_k, llm_provider):
    for _, row_idx in enumerate(unanswered_indices, start=1):
        question = df.at[row_idx, "question"]

        if getattr(config, "RAG_MODE", "STATIC") == "DYNAMIC":
            ans = dynamic_rag_answer(
                llm, doc_id, question, vector_store, llm_provider)
            df.at[row_idx, "llm_response"] = ans
            continue

        merged_chunks = load_vector_db_text(
            vector_store, doc_id, question, rag_top_k)
        messages = build_RAG_prompt(merged_chunks, question)

        if llm_provider == "Custom":
            response = llm.invoke(messages, testing_regime='RAG')
        else:
            response = llm.invoke(messages)

        df.at[row_idx, "llm_response"] = response.content.strip()


def process_gold_questions(llm, df, unanswered_indices, doc_chunks, llm_provider):
    """
    Process questions using GOLD (oracle baseline) approach.
    Uses pre-extracted document pieces guaranteed to contain answers.

    Args:
        llm: Language model instance
        df: DataFrame containing questions and responses
        unanswered_indices: List of row indices for unanswered questions
        doc_chunks: Document text (gold pieces)
        llm_provider: LLM provider name
    """
    for i, row_idx in enumerate(unanswered_indices, start=1):
        question = df.at[row_idx, "question"]
        messages = build_GOLD_prompt(doc_chunks, question)

        if llm_provider == "Custom":
            response = llm.invoke(messages, testing_regime='GOLD')
        else:
            response = llm.invoke(messages)

        df.at[row_idx, "llm_response"] = response.content.strip()


def process_single_chunk_questions(llm, df, unanswered_indices, doc_id,
                                   single_chunk_text, questions, context_chat,
                                   num_retries, wait_time_duration, model_name):
    """
    Process questions when document fits in a single chunk.
    Supports both single-question and batch modes.

    Args:
        llm: Language model instance
        df: DataFrame containing questions and responses
        unanswered_indices: List of row indices for unanswered questions
        doc_id: Document identifier
        single_chunk_text: Text of the single chunk
        questions: List of question texts
        context_chat: If True, process one question at a time; if False, batch all
        num_retries: Number of retry attempts
        wait_time_duration: Time between retries
        model_name: Name of the model
    """
    num_questions = len(questions)

    if context_chat:
        for i, row_idx in enumerate(unanswered_indices, start=1):
            question_text = df.at[row_idx, "question"]
            log_msg = f"[doc={doc_id} chunk=1 question_index={i}]"
            messages = build_prompt_single(single_chunk_text, question_text, i)
            parsed_answers = get_llm_json_response(
                llm, messages, 1, num_retries, wait_time_duration,
                model_name, extra_log_info=log_msg)
            df.at[row_idx, "llm_response"] = parsed_answers.get(
                1, "LLM parse error")
    else:
        log_msg = f"[doc={doc_id} chunk=1 batch_mode]"
        messages = build_prompt_batch(single_chunk_text, questions)
        parsed_answers = get_llm_json_response(
            llm, messages, num_questions, num_retries, wait_time_duration,
            model_name, extra_log_info=log_msg)
        for i, row_idx in enumerate(unanswered_indices, start=1):
            df.at[row_idx, "llm_response"] = parsed_answers.get(
                i, "LLM parse error")


def process_multi_chunk_questions_single_mode(llm, df, unanswered_indices, doc_id,
                                              doc_chunks, num_retries,
                                              wait_time_duration, model_name):
    """
    Process questions across multiple chunks in single-question mode.
    Each question is processed separately across all chunks, then combined.

    Args:
        llm: Language model instance
        df: DataFrame containing questions and responses
        unanswered_indices: List of row indices for unanswered questions
        doc_id: Document identifier
        doc_chunks: List of document chunk texts
        num_retries: Number of retry attempts
        wait_time_duration: Time between retries
        model_name: Name of the model
    """
    for i, row_idx in enumerate(unanswered_indices, start=1):
        question_text = df.at[row_idx, "question"]
        partial_responses = []

        for c_idx, chunk_text in enumerate(doc_chunks, start=1):
            log_msg = f"[doc={doc_id} chunk={c_idx} question_index={i}]"
            messages_chunk = build_prompt_single(chunk_text, question_text, i)
            chunk_parsed_answers = get_llm_json_response(
                llm, messages_chunk, 1, num_retries, wait_time_duration,
                model_name, extra_log_info=log_msg
            )

            answer_text = chunk_parsed_answers.get(1, "LLM parse error")
            if "LLM parse error" in answer_text:
                partial_responses.append(
                    '{"answers":[{"question_index":1,"answer":"LLM parse error"}]}')
            else:
                partial_json_str = json.dumps({
                    "answers": [
                        {"question_index": 1, "answer": answer_text}
                    ]
                })
                partial_responses.append(partial_json_str)

        combine_msg = f"[doc={doc_id} combine question_index={i}]"
        combine_prompt = build_prompt_combine_answers(
            partial_responses, [question_text])
        combined_final = get_llm_json_response(
            llm, combine_prompt, 1, num_retries, wait_time_duration,
            model_name, extra_log_info=combine_msg)
        df.at[row_idx, "llm_response"] = combined_final.get(
            1, "LLM parse error")


def process_multi_chunk_questions_batch_mode(llm, df, unanswered_indices, doc_id,
                                             doc_chunks, questions, num_retries,
                                             wait_time_duration, model_name):
    """
    Process questions across multiple chunks in batch mode.
    All questions are processed together for each chunk, then combined.

    Args:
        llm: Language model instance
        df: DataFrame containing questions and responses
        unanswered_indices: List of row indices for unanswered questions
        doc_id: Document identifier
        doc_chunks: List of document chunk texts
        questions: List of question texts
        num_retries: Number of retry attempts
        wait_time_duration: Time between retries
        model_name: Name of the model
    """
    num_questions = len(questions)
    partial_responses = []

    for c_idx, chunk_text in enumerate(doc_chunks, start=1):
        log_msg = f"[doc={doc_id} chunk={c_idx} batch_mode]"
        messages_chunk = build_prompt_batch(chunk_text, questions)
        chunk_parsed_answers = get_llm_json_response(
            llm, messages_chunk, num_questions, num_retries,
            wait_time_duration, model_name, extra_log_info=log_msg
        )

        if any(ans == "LLM parse error" for ans in chunk_parsed_answers.values()):
            fake_json = {
                "answers": [
                    {"question_index": i, "answer": "LLM parse error"}
                    for i in range(1, num_questions + 1)
                ]
            }
            partial_responses.append(json.dumps(fake_json))
        else:
            partial_json = {"answers": []}
            for q_idx in range(1, num_questions + 1):
                partial_json["answers"].append(
                    {"question_index": q_idx,
                     "answer": chunk_parsed_answers.get(q_idx, "LLM parse error")}
                )
            partial_responses.append(json.dumps(partial_json))

    combine_msg = f"[doc={doc_id} combine batch_mode]"
    combine_prompt = build_prompt_combine_answers(partial_responses, questions)
    combined_output = get_llm_json_response(
        llm, combine_prompt, num_questions, num_retries, wait_time_duration,
        model_name, extra_log_info=combine_msg)

    for i, row_idx in enumerate(unanswered_indices, start=1):
        df.at[row_idx, "llm_response"] = combined_output.get(
            i, "LLM parse error")


def process_document_questions(llm, df, unanswered_indices, doc_id, doc_chunks,
                               testing_regime, context_chat, vector_store,
                               rag_top_k, llm_provider, num_retries,
                               wait_time_duration, model_name, chunked_docs):
    """
    Main function to process all questions for a document.
    Routes to appropriate processing function based on regime and chunk count.

    Args:
        llm: Language model instance
        df: DataFrame containing questions and responses
        unanswered_indices: List of row indices for unanswered questions
        doc_id: Document identifier
        doc_chunks: List of document chunk texts (or single text for GOLD/RAG)
        testing_regime: 'FULL', 'RAG', or 'GOLD'
        context_chat: If True, single-question mode; if False, batch mode
        vector_store: FAISS vector store (for RAG)
        rag_top_k: Number of top chunks (for RAG)
        llm_provider: LLM provider name
        num_retries: Number of retry attempts
        wait_time_duration: Time between retries
        model_name: Name of the model
        chunked_docs: List to track documents that required chunking
    """
    questions = df.loc[unanswered_indices, "question"].tolist()

    if testing_regime == "RAG":
        process_rag_questions(llm, df, unanswered_indices, doc_id,
                              vector_store, rag_top_k, llm_provider)
        return

    if testing_regime == "GOLD":
        process_gold_questions(
            llm, df, unanswered_indices, doc_chunks, llm_provider)
        return

    if len(doc_chunks) == 1:
        process_single_chunk_questions(
            llm, df, unanswered_indices, doc_id, doc_chunks[0], questions,
            context_chat, num_retries, wait_time_duration, model_name)
    else:
        chunked_docs.append(doc_id)

        if context_chat:
            process_multi_chunk_questions_single_mode(
                llm, df, unanswered_indices, doc_id, doc_chunks,
                num_retries, wait_time_duration, model_name)
        else:
            process_multi_chunk_questions_batch_mode(
                llm, df, unanswered_indices, doc_id, doc_chunks, questions,
                num_retries, wait_time_duration, model_name)


def _parse_subqueries(raw: str) -> list[str]:
    """
    Extract {"subqueries":[...]} from raw LLM output (may be in ```json```).
    """
    m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    cleaned = m.group(1).strip() if m else raw.strip()

    try:
        data = json.loads(cleaned)
        subqs = data.get("subqueries", [])
        if not isinstance(subqs, list):
            return []
        return [str(x).strip() for x in subqs if str(x).strip()]
    except Exception:
        return []


def dynamic_rag_answer(llm, doc_id, question: str, vector_store, llm_provider: str) -> str:
    retrieved_docs = []

    for step in range(config.DYN_RAG_STEPS):
        # 1) planner
        planner_msgs = build_RAG_planner_prompt(
            question, config.DYN_RAG_QUERIES_PER_STEP
        )
        if llm_provider == "Custom":
            planner_resp = llm.invoke(
                planner_msgs, testing_regime="RAG").content.strip()
        else:
            planner_resp = llm.invoke(planner_msgs).content.strip()

        subqueries = _parse_subqueries(planner_resp)
        if not subqueries:
            subqueries = [question]

        # 2) retrieve per subquery
        for q in subqueries:
            docs = retrieve_vector_db_docs(
                vector_store, doc_id, q, config.DYN_RAG_TOP_K_PER_QUERY
            )
            retrieved_docs.extend(docs)

        # cheap stop condition: if we already have "enough" unique chunks, stop
        # (RAG_TOP_K is your typical retrieval budget)
        if len({d.metadata.get("chunk_id", None) for d in retrieved_docs}) >= config.RAG_TOP_K:
            break

    evidence = merge_and_dedupe_docs(
        retrieved_docs, config.DYN_RAG_MAX_EVIDENCE_CHARS)
    messages = build_RAG_prompt(evidence, question)

    if llm_provider == "Custom":
        response = llm.invoke(messages, testing_regime="RAG")
    else:
        response = llm.invoke(messages)

    return response.content.strip()
