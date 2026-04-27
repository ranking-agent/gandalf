"""Gandalf in-repo plugins.

Each plugin is a Python file in this package. Import side effects register
the plugin's hooks (filters, enrichers) into the shared registries. To add
a new plugin: create a file here, then add a `from . import <name>` line
below.

This is the single source of truth for what's installed.
"""

from . import enrichers  # noqa: F401  re-exported for convenience
from . import traversal_metadata_store  # noqa: F401
from . import max_node_degree  # noqa: F401
from . import information_content  # noqa: F401
from . import literature_cooccurrence_annotator  # noqa: F401
