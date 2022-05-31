import sys
from . import notebook_util as nbu

sys.meta_path.append(nbu.NotebookFinder())
