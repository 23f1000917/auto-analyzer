def log_problem_instance(p) -> None:
    print(
        f"{'=' * 100}\n"
        f"Problem Instance Created:\n"
        f"questions_text: {p.questions_text[:30]}...\n"
        f"attached_files: {p.filenames if p.filenames else None}\n"
        f"images: {len(p.images)}"
    )


def log_problem_metadata(metadata) -> None:
    print(
        f"{'=' * 100}\n"
        f"Problem Metadata Generated:\n"
        f"Data Source Description: \n"
        f"{metadata["data_source_desc"]}\n"
        f"Questions: \n"
        f"{"\n".join(f"- {q}" for q in metadata["questions"])}\n"
        f"Output Format Description: \n"
        f"{metadata["output_format_desc"]}\n"
    )


def log_final_files_dfs_script(script) -> None:
    print(f"{'=' * 100}\n" f"Final File Loading Script:\n" f"{script}")


def log_dfs_from_files(filenames, files_dfs) -> None:
    print(
        f"{'=' * 100}\n"
        f"{len(filenames)} attached files loaded as {len(files_dfs)} dataframes."
    )


def log_webscraped_tables(URL, scraped_dfs) -> None:
    print(
        f"{'=' * 100}\n" f"{len(scraped_dfs)} tables from {URL} loaded as dataframes.\n"
    )


def log_final_find_answer_script(question_str, script) -> None:
    print(f"{'=' * 100}\n" f"Final Script For '{question_str}':\n" f"{script}")


def log_script_attempt(purpose, attempt_no, error_message, fix_description) -> None:
    print(
        f"{'=' * 100}\n"
        f"Script For '{purpose}' Had Failed:\n"
        f"Attempt-{attempt_no}\n"
        f"Error: {error_message}\n"
        f"Applied Fix: {fix_description}",
    )
