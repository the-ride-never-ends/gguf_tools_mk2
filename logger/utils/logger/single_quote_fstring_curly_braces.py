import re
# TODO FIX THIS FUNCTION. IT DOESN'T WORK!!!!
def single_quote_fstring_curly_braces(msg: str) -> str:
    """
    NOTE Apparently this function won't work as there isn't a robust way to check if a string is an f-string or not after it's been processed.\n
        This is because the f-string syntax is evaluated at runtime, and by the time this function is called, 
        the string has already been processed and the f-string syntax has been removed.
        See: https://github.com/pylint-dev/pylint/issues/2507
    Add single quotes around curly braces in f-strings.

    This function processes a string that potentially contains f-string syntax.
    It adds single quotes around the content within curly braces, except when
    the curly braces are preceded by "\n" or ":".

    Args:
        msg (str): The input string to process.

    Returns:
        str: The processed string with single quotes added around f-string
             expressions, or the original string if it's not an f-string.

    Example:
        >>> name = 'Shrek'
        >>> statement_1 = f"{name} is love."
        >>> single_quote_fstring_curly_braces(statement_1)
        "'Shrek' is love."
        >>> name = 'Shrek'
        >>> statement_2 = f"\n{name} is life."
        >>> single_quote_fstring_curly_braces(statement_2)
        "\nShrek is life"
    """
    if isinstance(msg, str) and msg.startswith('f"'):
        def replacer(match):
            full_match = match.group(0)
            content = match.group(1)

            # Check if the curly brace is preceded by "\n" or ":"
            if match.start() > 0 and msg[match.start()-2:match.start()] in ("\n{", ": {"):
                return full_match

            return f"{{{content!r}}}"

        pattern = r'\{([^}]+?)\}'
        return re.sub(pattern, replacer, msg)
    return msg