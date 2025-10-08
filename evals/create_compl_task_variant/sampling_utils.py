import contextlib
import io
import numpy as np
import os
import pdb
import random
import tempfile
from pylint import run_pylint
from typing import List
from copy import deepcopy
import sys
import re
import pathlib

sys.path.insert(0, str(pathlib.Path().resolve()))
from evals.utils import get_diff


def convert_indentation(file_contents, target_indent=4, detect_only=False):
    """
    Convert indentation in a Python file while preserving nesting structure.

    Args:
        file_contents (str): The contents of the Python file as a string
        target_indent (int, optional): Number of spaces for target indentation. Defaults to 4.
        detect_only (bool, optional): If True, only returns detected indentation. Defaults to False.

    Returns:
        str: Converted file contents with new indentation, or detected indentation type
    """
    # Split the file into lines
    lines = file_contents.split("\n")

    # Detect current indentation type and size
    def detect_indentation(lines):
        indent_chars = []
        for line in lines:
            # Skip empty lines and comments
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue

            # Calculate indentation
            indent = line[: len(line) - len(stripped)]
            if indent and indent not in indent_chars:
                indent_chars.append(indent)

        # Return None if no indentation found
        if not indent_chars:
            return None

        # Prefer the most common indentation
        if "\t" in indent_chars:
            return "\t"

        # Find smallest space indentation
        space_indents = [ic for ic in indent_chars if ic.strip() == ""]
        if space_indents:
            return min(space_indents, key=len)

        return None

    # Detect current indentation
    current_indent = detect_indentation(lines)

    # If detect_only is True, return the detected indentation
    if detect_only:
        if current_indent == "\t\t":
            return "doubletab"
        elif current_indent == "\t":
            return "tab"
        elif current_indent:
            return len(current_indent)
        return None

    # If no indentation detected, return original contents
    if current_indent is None:
        return file_contents

    # Convert indentation
    converted_lines = []
    for line in lines:
        # Skip empty lines or lines with just whitespace
        if not line.strip():
            converted_lines.append("")
            continue

        # Determine current line's indentation
        current_line_indent = line[: len(line) - len(line.lstrip())]

        # Calculate new indentation depth
        if current_indent == "\t\t":
            # If currently using double tabs, convert tabs to equivalent spaces
            indent_depth = current_line_indent.count("\t\t")
        elif current_indent == "\t":
            # If currently using tabs, convert tabs to equivalent spaces
            indent_depth = current_line_indent.count("\t")
        else:
            # If using spaces, calculate current indent depth
            indent_depth = len(current_line_indent) // len(current_indent)

        # Create new indentation
        new_indentation = " " * (indent_depth * target_indent)

        # Replace existing indentation with new indentation
        converted_line = new_indentation + line.lstrip()
        converted_lines.append(converted_line)

    # Reconstruct file contents
    return "\n".join(converted_lines)


def detect_block_comments(python_code: str) -> List[int]:
    """
    Analyze Python code and return a binary array indicating which lines are part of block comments.

    Args:
        python_code (str): String containing Python code

    Returns:
        list[int]: Array where 1 indicates the line is within a block comment, 0 otherwise
    """
    lines = python_code.split("\n")
    result = [0] * len(lines)

    # Triple quotes that can start/end block comments
    block_comment_markers = ['"""', "'''"]

    in_block_comment = False
    current_marker = None

    for i, line in enumerate(lines):
        # Skip processing if we're in the middle of a string literal that contains what looks like a block comment
        if not in_block_comment:
            # Check if this line opens a block comment
            for marker in block_comment_markers:
                if marker in line:
                    # Check if the marker is in a string or is itself the start of a block comment
                    # This is a simple heuristic - a more robust solution would need a proper parser
                    pos = line.find(marker)
                    if pos > 0:
                        # Check if the marker is preceded by an odd number of backslashes (escaped)
                        backslash_count = 0
                        j = pos - 1
                        while j >= 0 and line[j] == "\\":
                            backslash_count += 1
                            j -= 1
                        if backslash_count % 2 == 1:
                            # Escaped marker, not a comment start
                            continue

                    # If line has an even number of the same marker, it's a single-line block comment
                    if line.count(marker) % 2 == 0 and line.count(marker) >= 2:
                        segments = line.split(marker)
                        # If we have more than one segment, check if there's code outside of the comments
                        if len(segments) > 2:
                            # Mark as a block comment line
                            result[i] = 1
                    else:
                        in_block_comment = True
                        current_marker = marker
                        result[i] = 1
                    break
        else:
            # We're inside a block comment, mark this line
            result[i] = 1

            # Check if this line closes the block comment
            if current_marker in line:
                # Again, this is a simple heuristic
                pos = line.find(current_marker)
                if pos >= 0:
                    # Check if the marker is escaped
                    backslash_count = 0
                    j = pos - 1
                    while j >= 0 and line[j] == "\\":
                        backslash_count += 1
                        j -= 1
                    if backslash_count % 2 == 0:  # Not escaped
                        in_block_comment = False
                        current_marker = None

    return result


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io(stream):
    """Capture outputs"""
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield stream


def file_linter(file, expected_error_traces=None):
    """
    Cleanly pylint contents of temp file. Treat identation warnings as errors.

    Return all (unexpected) error traces in standardized template formatting.
    """
    try:
        program_io = io.StringIO()
        with swallow_io(program_io):
            run_pylint(
                (
                    "--disable=C,R",
                    "--reports=n",
                    "--score=n",
                    "--unsafe-load-any-extension=y",
                    "--generated-members=cv2.*, hq.*, re.*",
                    "--msg-template={C}|{msg_id}|{line}|{column}|{msg}",
                    file,
                )
            )
    except BaseException as e:
        pass
    io_stream = program_io.getvalue()
    io_stream = io_stream.split("\n")
    error_traces = []
    for line in io_stream:
        try:
            C, msg_id, line_id, column_id, msg = line.split("|")
        except:
            continue

        # treat indentation warnings as errors
        if C == "W" and msg_id not in ["W0311", "W0231"]:
            continue
        elif msg_id == "E0001" and "on line" in msg:
            line_id = msg[: msg.rfind("(")].rstrip().split(" ")[-1]

        error_trace = (
            msg_id,
            line_id,
            column_id,
            msg,
        )
        if expected_error_traces is None or not error_trace in expected_error_traces:
            error_traces += [error_trace]
    return error_traces


def inflate_edit_path(
    code_as_text: str, edit_sequence: List[List[int]]
) -> (List[str], List[str]):
    """Apply the line indices of edits to the string representing the program/file.

    Return the corresponding code edits, expressed both as sequences of "raw" program
    states and as code "diffs".
    """
    lines = code_as_text.split("\n")

    integrated = []
    total_edit = []

    for step, edit in enumerate(edit_sequence):
        total_edit += edit
        integrated += [deepcopy(total_edit)]

    raw_text_seq = [code_as_text]
    for edit in integrated:
        out = "\n".join([line for (i, line) in enumerate(lines) if not i in edit])
        raw_text_seq += [out]

    del integrated
    del total_edit

    if raw_text_seq[-1] != "":
        raw_text_seq += [""]

    raw_text_seq = raw_text_seq[::-1]

    diff_text_seq = []

    for i in range(len(raw_text_seq) - 1):
        diff_text_seq += [get_diff(raw_text_seq[i], raw_text_seq[i + 1], strip=False)]
    return raw_text_seq, [d for d in diff_text_seq if len(d) > 0]


def random_chunked_trajectory(
    code_as_text: str,
    ignore_comments: bool = False,
    ignore_imports: bool = True,
    max_depth: int = 1000,
) -> List[List[int]]:
    lines = code_as_text.split("\n")
    code_as_text = convert_indentation(code_as_text, target_indent=4, detect_only=False)

    def _apply_deletion_edit(edit: List[int]):
        """
        Apply a deletion edit, represented as a list of line indices to be deleted.

        Returns:
            > A string representing the full program post deletion
            > A list of the remaining line indices in the program, post deletion
        """
        remaining_lines = set([i for i in range(len(lines))]).difference(set(edit))
        return "\n".join(
            [
                line
                for i, line in enumerate(code_as_text.split("\n"))
                if i in remaining_lines
            ]
        )

    candidate_lines = [i for i in range(len(lines))]

    if ignore_comments:
        block_comment_lines = detect_block_comments(code_as_text)
        candidate_lines = [
            i
            for i, (line, is_in_block) in enumerate(zip(lines, block_comment_lines))
            if i in candidate_lines
            and len(line.lstrip()) > 0
            and not "#" == line.lstrip()[0]
            and not is_in_block
        ]
    if ignore_imports:
        candidate_lines = [
            i
            for i, line in enumerate(lines)
            if i in candidate_lines and not "import" in line
        ]

    if len(candidate_lines) <= 2:
        return None

    edit_path = []
    has_errors = True
    depth = 1
    while has_errors:
        try:
            chunk_size = np.random.randint(
                1, high=len(candidate_lines) - 1
            )  ## leave at least one line!
        except:
            pdb.set_trace()
        chunk = random.sample(candidate_lines, chunk_size)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="r+") as fp:
            edited = _apply_deletion_edit(chunk)
            print(edited)
            fp.write(edited)
            fp.seek(0)
            error_traces = file_linter(fp.name)
            has_errors = len(error_traces) > 0
            fp.close()

        depth += 1

        if depth > max_depth:
            return None

    return [chunk, [i for i in lines if not i in chunk]]
