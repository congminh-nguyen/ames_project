TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Tuple, Union

    VERSION_TUPLE = Tuple[Union[int, str], ...]
else:
    VERSION_TUPLE = object

# Define version components
__version__: str
version_tuple: VERSION_TUPLE


# Set version dynamically
def set_version(version_str: str) -> None:
    global __version__, version_tuple
    __version__ = version_str
    version_tuple = tuple(
        int(part) if part.isdigit() else part for part in version_str.split(".")
    )


# Default version
set_version("0.0.1.post1")
