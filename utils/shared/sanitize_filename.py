
def sanitize_filename(input: str) -> str:
    """
    Sanitize a string to be used as (part of) a filename.
    """
    disallowed = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in disallowed:
        input = input.replace(char, ".")
    input = '.'.join(filter(None, input.split('.')))
    return input
