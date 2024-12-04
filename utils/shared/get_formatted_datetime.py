from datetime import datetime

def get_formatted_datetime():
    """
    Get the exact time in "%Y-%m-%d %H:%M:%S" format.

    Example
    >>> return get_formatted_datetime()
    '2024-09-11 11:13:00'
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

