import uuid

def make_id(return_bytes: bool = False) -> str | bytes:
    """
    Generate a unique identifier using UUID4.

    Args:
        return_bytes (bool, optional): If True, returns the UUID as bytes. 
                                       Defaults to False.
    Returns:
        str | bytes: A unique identifier as a string (default) or as bytes.
    """
    if return_bytes:
        id = uuid.uuid4()
        return id.bytes
    else:
        return str(uuid.uuid4())
