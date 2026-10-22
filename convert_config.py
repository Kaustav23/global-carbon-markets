import json
import re
from pathlib import Path


def parse_simple_yaml(text: str) -> dict:
  root = {}
  stack = [(root, -1)]

  for raw_line in text.splitlines():
    line = raw_line.split("#")[0].rstrip()
    if not line.strip():
      continue
    indent = len(line) - len(line.lstrip())
    content = line.strip()

    while stack and indent <= stack[-1][1]:
      stack.pop()
    curr = stack[-1][0]

    if content.startswith("- "):
      val_str = content[2:].strip()
      if isinstance(curr, list):
        if val_str.startswith("[") and val_str.endswith("]"):
          items = [
              float(x.strip())
              for x in val_str[1:-1].split(",")
              if x.strip()
          ]
          curr.append(items)
        else:
          curr.append(val_str.strip("\"'"))
      continue

    if ":" in content:
      key, val = content.split(":", 1)
      key = key.strip().strip("\"'")
      val = val.strip()

      if not val:
        new_container = [] if key in ("matrix", "hub_order") else {}
        curr[key] = new_container
        stack.append((new_container, indent))
      else:
        if val.lower() == "true":
          parsed = True
        elif val.lower() == "false":
          parsed = False
        elif re.match(r"^-?\d+$", val):
          parsed = int(val)
        elif re.match(r"^-?\d+\.\d+$", val):
          parsed = float(val)
        elif val.startswith("[") and val.endswith("]"):
          items = [x.strip().strip("\"'") for x in val[1:-1].split(",")]
          parsed = [
              float(x) if re.match(r"^-?\d+(\.\d+)?$", x) else x
              for x in items
              if x
          ]
        else:
          parsed = val.strip("\"'")
        curr[key] = parsed

  return root


yaml_path = Path("config/jurisdictions.yaml")
json_path = Path("config/jurisdictions.json")

data = parse_simple_yaml(yaml_path.read_text(encoding="utf-8"))
json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("[SUCCESS] Wrote config/jurisdictions.json")

# Update python references from yaml to json
files = [
    Path("src/ingestion/pipeline.py"),
    Path("src/mechanics/cap_vs_intensity.py"),
    Path("src/mechanics/stability_reserves.py"),
    Path("src/econometrics/price_discovery.py"),
    Path("src/econometrics/cbam_article6.py"),
    Path("app/streamlit_dashboard.py"),
]

for p in files:
  if p.exists():
    txt = p.read_text(encoding="utf-8")
    txt = txt.replace("import yaml", "import json")
    txt = txt.replace("yaml.safe_load(f)", "json.load(f)")
    txt = txt.replace('"jurisdictions.yaml"', '"jurisdictions.json"')
    txt = txt.replace("'jurisdictions.yaml'", "'jurisdictions.json'")
    p.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {p}")
    
