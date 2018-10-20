__version__ = '0.3.0.0'

# Either "tensorflow" or "torch"
backend = "tensorflow"
TF = "tensorflow"
TORCH = "torch"


def set_backend(backend_engine):
    """
    Set the backend engine, either "tensorflow" or "torch".
    This backend have to be set before importing anything else
    from the library
    Args:
        backend_engine (str): The backend engine
    """
    global backend
    backend = backend_engine
