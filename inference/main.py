import json
import langchain
import logging
import os
import pandas as pd
import re
import sys
import time

from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.cache import InMemoryCache
from pathlib import Path

import config

from model_loader import BaseModel

logging.basicConfig(level=logging.INFO)

langchain.llm_cache = InMemoryCache()


def clean_html(raw_html: str) -> str:
    """
    Minimal HTML-to-text cleaning with BeautifulSoup.
    Removes <script> and <style>, then collapses whitespace.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text


def chunk_text(doc_id: str, text: str, chunk_size: int):
    """
    Splits `text` into a list of substrings, each at most `chunk_size` characters.
    This is not for RAG, this is chunking in case of a large document.
    """
    if len(text) > chunk_size:
        logging.warning(
            f"Doc {doc_id} length {len(text)} > {chunk_size}, chunking..."
        )
        chunks = [text[i: i + chunk_size]
                  for i in range(0, len(text), chunk_size)]
        logging.info(
            f"Doc {doc_id} chunked into {len(chunks)} parts. Each up to {chunk_size} chars."
        )
        return chunks
    else:
        logging.info(
            f"Doc {doc_id} loaded, length={len(text)} chars."
        )
        return [text]


def load_document_text(doc_id: str, testing_regime: str = 'FULL') -> list[str]:
    """
    Load raw HTML and clean it, or load correct pieces for Oracle baseline.
    If longer than config.MAX_CHAR_FOR_SYSTEM, chunk it; else return as single chunk.
    Returns a list of chunks (one or more).
    """
    html_path = os.path.join(
        config.HTML_PATH, f'{doc_id}{".html" if testing_regime == "FULL" else "_gold.txt"}')
    if not os.path.exists(html_path):
        logging.error(f"HTML file not found: {html_path}")
        return []

    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            html_data = file.read()

        if testing_regime == 'GOLD':
            return html_data

        # FULL
        cleaned = clean_html(html_data)
        if not cleaned:
            logging.warning(f"Doc {doc_id} is empty after cleaning.")
            return []
        return chunk_text(doc_id, cleaned, config.MAX_CHAR_FOR_SYSTEM)

    except Exception as e:
        logging.error(f"Error reading {json_path}: {e}")
        return []


# def load_vector_db_text(vector_store, doc_id: str, question: str):
#     """
#     Extracting relevant chunks from a vector store for RAG baseline
#     """
#     retriever: VectorStoreRetriever = vector_store.as_retriever(
#         search_kwargs={"k": config.RAG_TOP_K, "filter": {"docID": str(doc_id)}}
#     )
#     results = retriever.get_relevant_documents(question)
#     result = "".join([doc.page_content for doc in results])
#     return result


# def build_RAG_prompt(document_text: str, question: str) -> list[dict]:
#     user_instructions = (
#         "[SYSTEM INPUT]\n"
#         "You are a financial expert, and your task is to answer "
#         "the question given to you based on the chunks of a credit agreement provided to you. "
#         "If you believe the answer is not present among the chunks, say 'Not found'.\n\n"

#         "[EXPECTED OUTPUT]\n"
#         "Respond ONLY with the answer to your question, nothing else. See the example below.\n\n"

#         "The given document:\n"
#         "Apple Inc. is a technology company headquartered in Cupertino, California. "
#         "It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.\n\n"

#         "The given question:\n"
#         "Where is the headquarters of Apple Inc.?\n\n"

#         "The expected output:\n"
#         "Cupertino, California\n\n"

#         "[USER INPUT]\n"
#         f"Merged chunks:\n{document_text}\n\n"

#         "[QUESTION]\n"
#         f"{question}\n"
#     )

#     return [{"role": "user", "content": user_instructions}]


# def build_GOLD_prompt(document_text: str, question: str) -> list[dict]:
#     user_instructions = (
#         "[SYSTEM INPUT]\n"
#         "You are a financial expert, and your task is to answer "
#         "the question given to you based on the pieces of a credit agreement which are guaranteed to contain the answer. "
#         "If you still believe the answer is not present among the chunks, say 'Not found'.\n\n"

#         "[EXPECTED OUTPUT]\n"
#         "Respond ONLY with the answer to your question, nothing else. See the example below.\n\n"

#         "The given document:\n"
#         "Apple Inc. is a technology company headquartered in Cupertino, California. "
#         "It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.\n\n"

#         "The given question:\n"
#         "Where is the headquarters of Apple Inc.?\n\n"

#         "The expected output:\n"
#         "Cupertino, California\n\n"

#         "[USER INPUT]\n"
#         f"Merged chunks:\n{document_text}\n\n"

#         "[QUESTION]\n"
#         f"{question}\n"
#     )

#     return [{"role": "user", "content": user_instructions}]


def build_prompt_single(document_text: str, question: str, question_index: int) -> list[dict]:
    """
    Single user message combining instructions + doc text + question.
    Enforce returning only JSON.
    """
    user_instructions = (
        "[SYSTEM INPUT]\n"
        "Your task is to answer questions given to you about the provided medical document. "
        "Forget all you general knowledge, provide answers based on the document content only. "
        "If you believe the answer is not present in the document, say 'Not found'.\n\n"

        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with valid JSON, nothing else. See the example below.\n\n"

        "The given document:\n"
        "Apple Inc. is a technology company headquartered in Cupertino, California. "
        "It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.\n\n"

        "The given question:\n"
        "Q1: What medication treats retinitis?\n\n"

        "The expected output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "azithromycin"}\n'
        "  ]\n"
        "}\n\n"

        "[USER INPUT]\n"
        f"Document:\n{document_text}\n\n"

        "[QUESTION]\n"
        f"Q{question_index}: {question}\n"
    )

    return [{"role": "user", "content": user_instructions}]


def build_prompt_batch(document_text: str, questions: list[str]) -> list[dict]:
    """
    Constructs a single user message that includes multiple questions about a credit agreement.
    The response must be in valid JSON format only.
    """
    prompt_lines = [
        "[SYSTEM INPUT]\n"
        "Your task is to answer questions given to you about the provided medical document. "
        "Forget all you general knowledge, provide answers based on the document content only. "
        "If you believe the answer is not present in the agreement, say 'Not found'.\n\n"

        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with valid JSON, nothing else. See the example below.\n\n"

        "The given document:\n"
        "Retinitis is caused by Bartonella and is treated by azithromycin.\n\n"

        "The given questions:\n"
        "Q1: What medication treats retinitis?\n"
        "Q2: What pathogen causes retinitis?\n\n"
        "Q1: What medication treats salmonellosis?\n"

        "The expected output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "azithromycin"},\n'
        '    {"question_index": 2, "answer": "bartonella"}\n'
        '    {"question_index": 3, "answer": "Not found"}\n'
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
    Build a prompt to merge/combine partial answers (in JSON form) from multiple chunks into final answers.
    We'll ask the LLM to output in the same JSON format:
    {
      "answers": [
        {"question_index": 1, "answer": "..."},
        ...
      ]
    }
    Enforce returning only JSON.
    """
    prompt_lines = [
        "[SYSTEM INPUT]\n"
        "Your task is to combine or merge the provided partial answers, coming from different chunks of a medical document, "
        "into a single final answer for each of the questions given to you. "
        "If you believe the answer is not present in the document, say 'Not found'. "
        "If there is a standard abbreviation (e.g., CDI for Clostridioides difficile–associated infection) "
        "for your answer, then use the abbreviation. Also, keep it general. For instance, if a specific "
        "pathogen has sub-types, then stick to the most general, popular pathogen. Same for diseases.\n\n"

        "[EXPECTED OUTPUT]\n"

        "The given document:\n"
        "Retinitis is caused by Bartonella and is treated by azithromycin.\n\n"

        "The given questions:\n"
        "Q1: What medication treats retinitis?\n"
        "Q2: What pathogen causes retinitis?\n\n"
        "Q1: What medication treats salmonellosis?\n"

        "The expected output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "azithromycin"},\n'
        '    {"question_index": 2, "answer": "bartonella"}\n'
        '    {"question_index": 3, "answer": "Not found"}\n'
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


def parse_llm_json(raw_response: str, num_questions: int) -> dict:
    """
    Expects a JSON string, possibly embedded in extra text, like:

    Some preamble text...
    ```json
    {
      "answers": [
        {"question_index": 1, "answer": "..."},
        ...
      ]
    }
    ```
    Some additional commentary...

    Returns a dict: {1: "answer1", 2: "answer2", ...}
    If invalid JSON or missing fields, defaults to "LLM parse error".
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


def parse_llm_list(raw_response: str, num_questions: int):
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


def call_llm_with_retries(llm, messages: list[dict], extra_log_info: str = "") -> str:
    """
    Call the LLM up to config.NUM_RETRIES times if blank is returned.
    We return the *raw string* from LLM (which should be JSON).
    extra_log_info can be used to log chunk/question context, etc.
    """
    for attempt in range(config.NUM_RETRIES):
        try:
            prompt_str = messages[0]['content'] if messages else ""
            logging.info(
                f"LLM call attempt {attempt+1}/{config.NUM_RETRIES} {extra_log_info} "
                f"(prompt length: {len(prompt_str)} chars)"
            )

            response = llm.invoke(messages)
            raw_output = response.content.strip() if hasattr(
                response, "content") else str(response).strip()
            print(raw_output)
            time.sleep(10.0)

            if raw_output:
                if config.WAIT_TIME_ENABLED:
                    time.sleep(config.WAIT_TIME_DURATION)
                return raw_output
            else:
                logging.warning(
                    f"Got an empty response from LLM. Retrying in 1s...")
                time.sleep(1.0)
        except Exception as e:
            print(raw_output)
            logging.error(f"LLM call error on attempt {attempt+1}: {e}")
            time.sleep(1.0)

    logging.error("All attempts returned empty response. Giving up.")
    return ""


def get_llm_json_response(llm, messages: list[dict], num_questions: int, extra_log_info: str) -> dict:
    """
    Attempts to get a valid JSON parse from the LLM.
    Retries multiple times (config.NUM_RETRIES) if the JSON parse fails.
    """
    parsed_result = {}
    for parse_attempt in range(config.NUM_RETRIES):
        raw_output = call_llm_with_retries(
            llm, messages, extra_log_info=extra_log_info)
        if not raw_output:
            logging.warning(
                f"Empty output from LLM (parse attempt {parse_attempt+1}). Retrying...")
            time.sleep(1.0)
            continue

        parsed_result = parse_llm_json(raw_output, num_questions)
        # Check if parse was successful
        if any(ans != "LLM parse error" for ans in parsed_result.values()):
            return parsed_result
        else:
            if config.MODEL_NAME != 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo':
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

    return parsed_result


def main():
    # 1) Load the model
    model_loader = BaseModel(
        llm_provider=config.LLM_PROVIDER,
        model_name=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        max_tokens=config.max_tokens_generation
    )
    model_loader.load()
    llm = model_loader.get_model()
    # if config.TESTING_REGIME == "RAG":
    #     embedding_model = HuggingFaceEmbeddings(model_name=config.RAG_MODEL)
    #     vector_store = FAISS.load_local(
    #         config.VECTOR_DB_DIR, embeddings=embedding_model,
    #         allow_dangerous_deserialization=True)

    # 2) Read CSVs
    directory = Path(config.INPUT_PATH)
    csv_files = list(directory.glob("*.csv"))
    for input_csv_path in csv_files:
        # input_csv_path = os.path.join(
        #     config.INPUT_PATH, f"{config.QUESTION_FILE}.csv")
        # if not os.path.exists(input_csv_path):
        #     logging.error(f"Input CSV not found: {input_csv_path}")
        #     return

        df = pd.read_csv(input_csv_path)
        if len(df) == 0:
            continue
        print(f'\n\n\n\n\n{input_csv_path}\n\n\n\n\n')

        # Ensure we have an "llm_response" column to store results
        if "llm_response" not in df.columns:
            df["llm_response"] = ""

        # Group by document_number
        grouped = df.groupby("document_number")
        all_doc_ids = list(grouped.groups.keys())

        total_docs = len(all_doc_ids)
        processed_docs = 0
        chunked_docs = []

        logging.info(f"Total documents to process: {total_docs}")

        output_dir = config.OUTPUT_PATH
        os.makedirs(output_dir, exist_ok=True)

        def save_csv():
            """
            Saves the current DataFrame to CSV, then runs evaluation metrics.
            """
            # 4) Save results
            sanitized_model_name = config.MODEL_NAME.replace("/", "-")
            sanitized_model_name = re.sub(
                r'[<>:"/\\|?*]', '-', sanitized_model_name)

            # output_csv = f"{config.QUESTION_FILE}_{sanitized_model_name}_{config.TESTING_REGIME}.csv"
            output_csv = f"{input_csv_path.stem}_{sanitized_model_name}_{config.TESTING_REGIME}.csv"
            output_path = os.path.join(output_dir, output_csv)
            df.to_csv(output_path, index=False)
            logging.info(f"Saved LLM answers to {output_path}")

        try:
            # NEW OR CHANGED: We wrap the main doc-loop in a try/except
            for doc_id in all_doc_ids:
                processed_docs += 1
                logging.info(
                    f"Processing document {processed_docs}/{total_docs} (doc_id={doc_id})...")

                group_indices = grouped.groups[doc_id]
                overall_indices_list = list(group_indices)

                question_batch_length = 50
                doc_chunks = load_document_text(
                    str(doc_id), config.TESTING_REGIME)  # str for "GOLD"

                # If doc text is empty, mark all as 'No doc text'
                if not doc_chunks:
                    logging.warning(
                        f"Document {doc_id} is empty. Setting llm_response='No doc text'.")
                    for idx in overall_indices_list:
                        df.at[idx, "llm_response"] = "No doc text"
                    # Save partial results after finishing each doc
                    save_csv()
                    continue

                # We process the doc's questions in sub-batches
                for q_start in range(0, len(overall_indices_list), question_batch_length):
                    q_indices_list = overall_indices_list[q_start: q_start +
                                                          question_batch_length]
                    questions = df.loc[q_indices_list, "question"].tolist()
                    num_questions = len(questions)
                    logging.info(
                        f"Processing {num_questions} questions for doc_id={doc_id}...")

                    # if config.TESTING_REGIME == "RAG":
                    #     for i, row_idx in enumerate(q_indices_list, start=1):
                    #         question = df.at[row_idx, "question"]
                    #         merged_chunks = load_vector_db_text(
                    #             vector_store, doc_id, question)
                    #         messages = build_RAG_prompt(merged_chunks, question)
                    #         if config.LLM_PROVIDER == "Custom":
                    #             response = llm.invoke(
                    #                 messages, testing_regime=config.TESTING_REGIME)
                    #         else:
                    #             response = llm.invoke(messages)
                    #         df.at[row_idx, "llm_response"] = response.content.strip()
                    #     continue

                    # if config.TESTING_REGIME == "GOLD":
                    #     for i, row_idx in enumerate(q_indices_list, start=1):
                    #         question = df.at[row_idx, "question"]
                    #         messages = build_GOLD_prompt(doc_chunks, question)
                    #         if config.LLM_PROVIDER == "Custom":
                    #             response = llm.invoke(
                    #                 messages, testing_regime=config.TESTING_REGIME)
                    #         else:
                    #             response = llm.invoke(messages)
                    #         df.at[row_idx, "llm_response"] = response.content.strip()
                    #     continue

                    # If there's only 1 chunk, process it normally
                    if len(doc_chunks) == 1:
                        single_chunk_text = doc_chunks[0]
                        if config.context_chat:
                            for i, row_idx in enumerate(q_indices_list, start=1):
                                question_text = df.at[row_idx, "question"]
                                log_msg = f"[doc={doc_id} chunk=1 question_index={i}]"
                                messages = build_prompt_single(
                                    single_chunk_text, question_text, i)
                                parsed_answers = get_llm_json_response(
                                    llm, messages, 1, extra_log_info=log_msg)
                                df.at[row_idx, "llm_response"] = parsed_answers[1]
                        else:
                            log_msg = f"[doc={doc_id} chunk=1 batch_mode]"
                            messages = build_prompt_batch(
                                single_chunk_text, questions)
                            parsed_answers = get_llm_json_response(
                                llm, messages, num_questions, extra_log_info=log_msg)
                            for i, row_idx in enumerate(q_indices_list, start=1):
                                df.at[row_idx, "llm_response"] = parsed_answers[i]

                    else:
                        # Document required chunking, add to tracking
                        chunked_docs.append(doc_id)

                        # If multiple chunks
                        if config.context_chat:
                            # Each question is separate across all chunks
                            for i, row_idx in enumerate(q_indices_list, start=1):
                                question_text = df.at[row_idx, "question"]
                                partial_responses = []

                                for c_idx, chunk_text in enumerate(doc_chunks, start=1):
                                    log_msg = f"[doc={doc_id} chunk={c_idx} question_index={i}]"
                                    messages_chunk = build_prompt_single(
                                        chunk_text, question_text, i)
                                    chunk_parsed_answers = get_llm_json_response(
                                        llm, messages_chunk, 1, extra_log_info=log_msg
                                    )

                                    if "LLM parse error" in chunk_parsed_answers[1]:
                                        partial_responses.append(
                                            '{"answers":[{"question_index":1,"answer":"LLM parse error"}]}')
                                    else:
                                        partial_json_str = json.dumps({
                                            "answers": [
                                                {"question_index": 1,
                                                    "answer": chunk_parsed_answers[1]}
                                            ]
                                        })
                                        partial_responses.append(
                                            partial_json_str)

                                combine_msg = f"[doc={doc_id} combine question_index={i}]"
                                combine_prompt = build_prompt_combine_answers(
                                    partial_responses, [question_text])
                                combined_final = get_llm_json_response(
                                    llm, combine_prompt, 1, extra_log_info=combine_msg)
                                df.at[row_idx, "llm_response"] = combined_final[1]

                        else:
                            # Batch mode across multiple chunks
                            partial_responses = []
                            for c_idx, chunk_text in enumerate(doc_chunks, start=1):
                                log_msg = f"[doc={doc_id} chunk={c_idx} batch_mode]"
                                messages_chunk = build_prompt_batch(
                                    chunk_text, questions)
                                chunk_parsed_answers = get_llm_json_response(
                                    llm, messages_chunk, num_questions, extra_log_info=log_msg
                                )

                                if any(ans == "LLM parse error" for ans in chunk_parsed_answers.values()):
                                    fake_json = {
                                        "answers": [
                                            {"question_index": i,
                                                "answer": "LLM parse error"}
                                            for i in range(1, num_questions + 1)
                                        ]
                                    }
                                    partial_responses.append(
                                        json.dumps(fake_json))
                                else:
                                    partial_json = {"answers": []}
                                    for q_idx in range(1, num_questions + 1):
                                        partial_json["answers"].append(
                                            {"question_index": q_idx,
                                                "answer": chunk_parsed_answers[q_idx]}
                                        )
                                    partial_responses.append(
                                        json.dumps(partial_json))

                            combine_msg = f"[doc={doc_id} combine batch_mode]"
                            combine_prompt = build_prompt_combine_answers(
                                partial_responses, questions)
                            combined_output = get_llm_json_response(
                                llm, combine_prompt, num_questions, extra_log_info=combine_msg)
                            for i, row_idx in enumerate(q_indices_list, start=1):
                                df.at[row_idx, "llm_response"] = combined_output[i]

                logging.info(
                    f"Completed processing document {processed_docs}/{total_docs} (doc_id={doc_id}).")
                # NEW OR CHANGED: partial saving after each doc
                save_csv()

        except KeyboardInterrupt:
            # NEW OR CHANGED: If the user presses Ctrl+C or otherwise interrupts
            logging.warning(
                "Code terminated by user. Marking unprocessed documents with 'code terminated'...")

            # Mark all documents not processed yet as "code terminated"
            # processed_docs is the count of docs we already did
            # The ones we haven't started
            unprocessed_docs = all_doc_ids[processed_docs:]
            for udoc_id in unprocessed_docs:
                indices_ = grouped.groups[udoc_id]
                for idx in indices_:
                    if df.at[idx, "llm_response"] == "":
                        df.at[idx, "llm_response"] = "code terminated"

            # Save final partial results
            save_csv()
            logging.warning("Partial results saved. Exiting now.")
            sys.exit(1)

        except Exception as e:
            # If there's an unexpected exception, log it
            logging.error(f"Unexpected top-level error: {e}", exc_info=True)

            # Mark all documents not processed as "code terminated"
            unprocessed_docs = all_doc_ids[processed_docs:]
            for udoc_id in unprocessed_docs:
                indices_ = grouped.groups[udoc_id]
                for idx in indices_:
                    if df.at[idx, "llm_response"] == "":
                        df.at[idx, "llm_response"] = "code terminated"

            # Save partial results
            save_csv()
            logging.warning(
                "Partial results saved. Exiting due to fatal error.")
            sys.exit(1)

        # If we complete everything without interruption:
        logging.info(
            f"Processing complete: {processed_docs}/{total_docs} documents processed successfully.")

        # 6) Summarize chunked docs
        if chunked_docs:
            unique_chunked = list(set(chunked_docs))
            logging.info(
                f"Documents that required chunking: {len(unique_chunked)}")
            logging.info(f"Chunked Document IDs: {unique_chunked}")
            print("\nDocuments that required chunking:", unique_chunked)
        else:
            logging.info("No documents required chunking.")


if __name__ == "__main__":
    main()
