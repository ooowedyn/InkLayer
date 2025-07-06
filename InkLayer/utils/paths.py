import os
import InkLayer

def get_model_path(filename):
    ink_root = os.path.dirname(InkLayer.__file__)
    return os.path.join(ink_root, "..", "models", filename)
