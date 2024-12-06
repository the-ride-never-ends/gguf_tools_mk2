from datetime import datetime

def _right_now() -> str:
    """Return current datetime as formatted string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")