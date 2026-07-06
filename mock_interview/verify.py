# Dev-only: run every stage's model solution against its own hidden tests.
# All green = the problem bank is internally consistent.
#
#   python3 verify.py

from app import grade
from problems import SESSIONS

IMPORT = "import numpy as np\n\n"

total = failed = 0


def check(label, code, tests):
    global total, failed
    r = grade(code, tests)
    if not r["ok"]:
        failed += 1
        total += 1
        print(f"CRASH  {label}")
        print(r["stderr"][-800:])
        return
    for t in r["results"]:
        total += 1
        if t["ok"]:
            print(f"pass   {label}  {t['name']}")
        else:
            failed += 1
            print(f"FAIL   {label}  {t['name']}")
            print("       " + t["msg"][:400])


for sess in SESSIONS:
    acc = IMPORT  # base solutions accumulate within a session, like a live code buffer
    for st in sess["stages"]:
        # each variant runs against the accumulated PRIOR stages + its own solution
        for vi, var in enumerate(st.get("variants", [])):
            check(f"{sess['id']}/{st['id']}[v{vi + 1}:{var['label']}]",
                  acc + var["solution"] + "\n", var["tests"])
        acc += st["solution"] + "\n\n"
        check(f"{sess['id']}/{st['id']}", acc, st["tests"])

print(f"\n{total - failed}/{total} checks passed")
raise SystemExit(1 if failed else 0)
