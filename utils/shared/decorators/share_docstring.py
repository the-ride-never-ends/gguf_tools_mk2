
from functools import wraps

def share_docstring(template_method):
    """
    A decorator that shares the docstring of a template method with the decorated method.

    This decorator is designed to be used in class definitions. It copies the docstring
    from a specified template method to the decorated method at runtime.

    Args:
        template_method (str): The name of the method whose docstring should be shared.

    Returns:
        function: A decorator function that can be applied to class methods.

    Example:
    >>> class MyClass:
    >>>     def template(self):
    >>>         '''This is the template docstring.'''
    >>>         pass
    >>> 
    >>>     @share_docstring('template')
    >>>     def decorated_method(self):
    >>>         # This method will inherit the docstring from 'template'
    >>>         pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get the template method from the instance
            template = getattr(self, template_method)
            # Share the docstring
            wrapper.__doc__ = template.__doc__
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
