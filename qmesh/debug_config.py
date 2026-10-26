import sys, time
sys.path.insert(0, "/src/exllamav3")
print("import", flush=True)
from exllamav3 import Config, Model
print("fromdir", flush=True)
c = Config.from_directory("/cold")
print("config", c.architecture, flush=True)
m = Model.from_config(c)
print("model", len(m.modules), flush=True)
