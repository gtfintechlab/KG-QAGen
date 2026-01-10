"""
Main inference pipeline for LLM-based question answering on credit agreements.
This module orchestrates the entire workflow: loading data, processing questions,
calling the LLM, and saving results.
"""

import langchain
import logging
import os
import pandas as pd
import re
import sys

import config

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.cache import InMemoryCache

from model_loader import BaseModel
from document_processor import load_document_text
from question_processor import process_document_questions

logging.basicConfig(level=logging.INFO)
langchain.llm_cache = InMemoryCache()


def is_question_answered(response) -> bool:
    """
    Check if a question has already been answered.
    A question is considered answered if the response is not empty,
    not "code terminated", and not NaN.

    Args:
        response: Response value from DataFrame (may be string, array, or NaN)

    Returns:
        True if question is answered, False otherwise
    """
    if hasattr(response, '__iter__') and not isinstance(response, str):
        try:
            response = response.iloc[0] if hasattr(
                response, 'iloc') else response[0]
        except (IndexError, KeyError):
            return False

    try:
        if pd.isna(response):
            return False
    except (ValueError, TypeError):
        return False

    if isinstance(response, str):
        response = response.strip()
        if response == "" or response.lower() == "code terminated":
            return False
        return True
    return False


def load_existing_results(output_path, df):
    """
    Load existing results from a previous run if the output file exists.

    Args:
        output_path: Path to the output CSV file
        df: DataFrame to populate with existing results

    Returns:
        Updated DataFrame with existing responses loaded
    """
    if os.path.exists(output_path):
        logging.info(f"Found existing results file: {output_path}")
        existing_df = pd.read_csv(output_path)
        if "llm_response" in existing_df.columns:
            df["llm_response"] = existing_df["llm_response"]
            logging.info(
                "Loaded existing responses. Will skip already-answered questions.")
    return df


def save_results(df, output_dir, question_file, model_name, testing_regime):
    """
    Save the current DataFrame to CSV.

    Args:
        df: DataFrame containing questions and responses
        output_dir: Directory to save results
        question_file: Name of the question file (without extension)
        model_name: Name of the model
        testing_regime: Testing regime (FULL, RAG, or GOLD)
    """
    sanitized_model_name = model_name.replace("/", "-")
    sanitized_model_name = re.sub(r'[<>:"/\\|?*]', '-', sanitized_model_name)
    output_csv = f"{question_file}_{sanitized_model_name}_{testing_regime}.csv"
    output_path = os.path.join(output_dir, output_csv)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved LLM answers to {output_path}")


def log_processing_stats(df, grouped):
    """
    Log statistics about the processing job.

    Args:
        df: DataFrame containing questions
        grouped: Grouped DataFrame by document_number
    """
    total_docs = len(list(grouped.groups.keys()))
    total_questions = len(df)
    answered_questions = sum(
        1 for idx in df.index if is_question_answered(df.at[idx, "llm_response"]))
    unanswered_questions = total_questions - answered_questions

    logging.info(f"Total documents to process: {total_docs}")
    logging.info(f"Total questions: {total_questions}")
    logging.info(f"Already answered: {answered_questions}")
    logging.info(f"To be processed: {unanswered_questions}")

    return total_docs, unanswered_questions


def mark_unanswered_as_terminated(df):
    """
    Mark all unanswered questions as 'code terminated'.
    Used when processing is interrupted.

    Args:
        df: DataFrame containing questions and responses
    """
    for idx in df.index:
        if not is_question_answered(df.at[idx, "llm_response"]):
            df.at[idx, "llm_response"] = "code terminated"


def main():
    """
    Main execution function for the inference pipeline.
    """
    # 1) Load the model
    logging.info("Initializing model...")
    model_loader = BaseModel(
        llm_provider=config.LLM_PROVIDER,
        model_name=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        max_tokens=config.max_tokens_generation
    )
    model_loader.load()
    llm = model_loader.get_model()

    # Load vector store if using RAG
    vector_store = None
    if config.TESTING_REGIME == "RAG":
        logging.info("Loading vector store for RAG...")
        embedding_model = HuggingFaceEmbeddings(model_name=config.RAG_MODEL)
        vector_store = FAISS.load_local(
            config.VECTOR_DB_DIR, embeddings=embedding_model,
            allow_dangerous_deserialization=True)

    # 2) Read the input CSV
    input_csv_path = os.path.join(
        config.INPUT_PATH, f"{config.QUESTION_FILE}.csv")
    if not os.path.exists(input_csv_path):
        logging.error(f"Input CSV not found: {input_csv_path}")
        return

    logging.info(f"Loading questions from {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    if "llm_response" not in df.columns:
        df["llm_response"] = ""

    # 3) Check for existing results and load them
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)

    sanitized_model_name = config.MODEL_NAME.replace("/", "-")
    sanitized_model_name = re.sub(r'[<>:"/\\|?*]', '-', sanitized_model_name)
    output_csv = f"{config.QUESTION_FILE}_{sanitized_model_name}_{config.TESTING_REGIME}{'_' + config.RAG_MODE if config.TESTING_REGIME == 'RAG' else ''}.csv"
    output_path = os.path.join(output_dir, output_csv)

    df = load_existing_results(output_path, df)

    # 4) Group by document and log statistics
    grouped = df.groupby("document_number")
    all_doc_ids = list(grouped.groups.keys())
    total_docs, unanswered_questions = log_processing_stats(df, grouped)

    if unanswered_questions == 0:
        logging.info("All questions are already answered. Nothing to process.")
        return

    processed_docs = 0
    chunked_docs = []

    try:
        # 5) Process each document
        for doc_id in all_doc_ids:
            processed_docs += 1
            logging.info(
                f"Processing document {processed_docs}/{total_docs} (doc_id={doc_id})...")

            group_indices = grouped.groups[doc_id]
            overall_indices_list = list(group_indices)

            all_answered = all(
                is_question_answered(df.at[idx, "llm_response"])
                for idx in overall_indices_list
            )

            if all_answered:
                logging.info(
                    f"All questions for document {doc_id} are already answered. Skipping."
                )
                continue

            question_batch_length = 50
            doc_chunks = load_document_text(
                str(doc_id), config.HTML_PATH, config.MAX_CHAR_FOR_SYSTEM,
                config.TESTING_REGIME)

            if not doc_chunks:
                logging.warning(
                    f"Document {doc_id} is empty. Setting llm_response='No doc text'.")
                for idx in overall_indices_list:
                    if not is_question_answered(df.at[idx, "llm_response"]):
                        df.at[idx, "llm_response"] = "No doc text"
                save_results(df, output_dir, config.QUESTION_FILE,
                             config.MODEL_NAME, f"{config.TESTING_REGIME}{'_' + config.RAG_MODE if config.TESTING_REGIME == 'RAG' else ''}")
                continue

            for q_start in range(0, len(overall_indices_list), question_batch_length):
                q_indices_list = overall_indices_list[q_start: q_start +
                                                      question_batch_length]

                unanswered_indices = [
                    idx for idx in q_indices_list
                    if not is_question_answered(df.at[idx, "llm_response"])
                ]

                if not unanswered_indices:
                    logging.info(
                        f"All questions in this batch for doc_id={doc_id} are already answered. Skipping batch."
                    )
                    continue

                num_questions = len(unanswered_indices)
                logging.info(
                    f"Processing {num_questions} unanswered questions for doc_id={doc_id}..."
                )

                process_document_questions(
                    llm=llm,
                    df=df,
                    unanswered_indices=unanswered_indices,
                    doc_id=doc_id,
                    doc_chunks=doc_chunks,
                    testing_regime=config.TESTING_REGIME,
                    context_chat=config.context_chat,
                    vector_store=vector_store,
                    rag_top_k=config.RAG_TOP_K,
                    llm_provider=config.LLM_PROVIDER,
                    num_retries=config.NUM_RETRIES,
                    wait_time_duration=config.WAIT_TIME_DURATION,
                    model_name=config.MODEL_NAME,
                    chunked_docs=chunked_docs
                )

            logging.info(
                f"Completed processing document {processed_docs}/{total_docs} (doc_id={doc_id}).")

            save_results(df, output_dir, config.QUESTION_FILE,
                         config.MODEL_NAME, f"{config.TESTING_REGIME}{'_' + config.RAG_MODE if config.TESTING_REGIME == 'RAG' else ''}")

    except KeyboardInterrupt:
        logging.warning(
            "Code terminated by user. Marking unprocessed questions with 'code terminated'...")
        mark_unanswered_as_terminated(df)
        save_results(df, output_dir, config.QUESTION_FILE,
                     config.MODEL_NAME, f"{config.TESTING_REGIME}{'_' + config.RAG_MODE if config.TESTING_REGIME == 'RAG' else ''}")
        logging.warning("Partial results saved. Exiting now.")
        sys.exit(1)

    except Exception as e:
        logging.error(f"Unexpected top-level error: {e}", exc_info=True)
        mark_unanswered_as_terminated(df)
        save_results(df, output_dir, config.QUESTION_FILE,
                     config.MODEL_NAME, f"{config.TESTING_REGIME}{'_' + config.RAG_MODE if config.TESTING_REGIME == 'RAG' else ''}")
        logging.warning("Partial results saved. Exiting due to fatal error.")
        sys.exit(1)

    # 6) Final summary
    logging.info(
        f"Processing complete: {processed_docs}/{total_docs} documents processed successfully.")

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
