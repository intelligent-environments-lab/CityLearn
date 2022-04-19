import importlib
module = importlib.import_module('agent')
class_ = getattr(module,'Agent')
print(class_)
assert False