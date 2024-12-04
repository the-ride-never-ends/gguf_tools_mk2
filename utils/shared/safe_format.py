
import string

class SafeFormatter(string.Formatter):
    """
    A custom string formatter that safely handles missing keys and invalid format strings.
    
    This class extends the built-in string.Formatter to provide more robust formatting
    capabilities, particularly when dealing with potentially missing keys or malformed
    format strings.
    """

    def get_value(self, key, args, kwargs):
        """
        Retrieve a value for a given key from kwargs, or return a placeholder if not found.
        Args:
            key: The key to look up in kwargs.
            args: Positional arguments (not used in this implementation).
            kwargs: Keyword arguments containing the values to format.

        Returns:
            The value associated with the key if found in kwargs, otherwise a placeholder
            string containing the key.
        """
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        else:
            return super().get_value(key, args, kwargs)

    def parse(self, format_string):
        """
        Parse the format string, handling potential ValueError exceptions.

        Args:
            format_string: The string to be parsed for formatting.

        Returns:
            A list of tuples representing the parsed format string. If parsing fails,
            returns a list with a single tuple containing the entire format string.
        """
        try:
            return super().parse(format_string)
        except ValueError:
            return [(format_string, None, None, None)]


def safe_format(format_string: str, *args, **kwargs) -> str:
    """


    Example:
    >>> kwargs = {
    >>>     "first": 1
    >>>     "second": "second"
    >>>     "third": get_bool("three")
    >>>     "fourth": 2*2
    >>> }
    >>> with open("text.txt", "r") as file:
    >>>     string = file.read()
    >>> return string
    "{first}, {second}, {third}, {fourth}, fifth"
    >>> string = safe_format(string, **kwargs)
    >>> return string
    "1, second, True, 4, fifth"

    Args:
        format_string: The string to be formatted.
        *args: Variable length argument list (not used in the current implementation).
        **kwargs: Arbitrary keyword arguments to be used in formatting.

    Returns:
        A formatted string where keys from kwargs are substituted into the format_string.
        Missing keys are left as is in the resulting string.
    """
    formatter = SafeFormatter()
    return formatter.format(format_string, *args, **kwargs)
