from datetime import datetime

def convert_integer_to_datetime_str(integer: int) -> str:
    """
    Converts an integer representation of a datetime to a formatted string.

    This function takes an integer input representing a datetime in the format
    YYYYMMDDhhmmss and converts it to a string in the format 'YYYY-MM-DD HH:MM:SS'.

    Args:
        integer (int): An integer representing a datetime in the format YYYYMMDDhhmmss.

    Returns:
        str: A string representing the datetime in the format 'YYYY-MM-DD HH:MM:SS'.

    Raises:
        AssertionError: If the input integer, when converted to a string, is not exactly 14 characters long.

    Example:
    >>> integer_input = 20240802121308
    >>> result = convert_integer_to_datetime_str(integer_input)
    >>> print(result)
    '2024-08-02 12:13:08'
    """

    # Convert the integer to a string
    date_string = str(integer)
    assert len(date_string) == 14, f"len(date_string) is not 14, but '{len(date_string)}', so it cannot be converted to YYYY-MM-DD hh:mm:ss format"
    
    # Parse the string into a datetime object
    dt = datetime.strptime(date_string, "%Y%m%d%H%M%S")
    
    # Format the datetime object into the desired string format
    formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return formatted_date
