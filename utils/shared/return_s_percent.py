
def return_s_percent(args: dict) -> str|None:
    """
    Produce a comma-separated string of "%s" based on the length of the input.
    If it's unable to parse the length of the input, returns None.

    Example
        >>> return_s_percent(3)
        '%s,%s,%s'
        >>> return_s_percent([1, 2, 3, 4])
        '%s,%s,%s,%s'
        >>> return_s_percent("hello")
        '%s,%s,%s,%s,%s'
        >>> return_s_percent(None)
        ''
    """
    try:
        return ",".join(["%s"] * len(args))
    except:
        return None

