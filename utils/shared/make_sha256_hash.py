import hashlib

def make_sha256_hash(*args) -> str:
    """
    Generate a SHA-256 hash from any positional arguments.
    
    Takes any number of arguments, concatenates them into a one string,
    and then generates a SHA-256 hash of the resulting string.

    NOTE: The order of the args tuple changes the output hash. 

    NOTE: url_hash = gnis + url, NOT url + gnis

    Args:
        args: Variable length argument tuple. Can be any type that can be converted to a string.

    ## Example
    >>> return make_sha256_hash("hello", "world", 123)
    '98d234d5303d20f5f757b1f813907dac130e7066cc881e1c2bcd9feb398ba68b'
    """
    # Turn the args tuple into a list
    args = list(args)

    # Turn all the elements in the list to strings, then smash them all together.
    args = "".join(str(arg) for arg in args)

    # Return a string representation of a SHA256 hash. 
    return hashlib.sha256(bytes(args, encoding="utf-8")).hexdigest()
