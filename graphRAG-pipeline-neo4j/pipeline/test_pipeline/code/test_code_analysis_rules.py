"""Pure tests for code_analysis rules: no Neo4j, no LLM."""
import pytest

from pipeline.code_analysis.models import Finding
from pipeline.code_analysis.rules import RULE_INFO, scan_source
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON


def _rule_ids(source, file_name="app.py"):
    return [f.rule_id for f in scan_source(source, file_name)]


# One tiny vulnerable snippet per rule: (rule id, source).
VULNERABLE = [
    ("eval-exec", "def f(x):\n    return eval(x)\n"),
    ("shell-command", "import os\n\ndef f(name):\n    os.system('ls ' + name)\n"),
    ("shell-command", "import subprocess\n\ndef f(cmd):\n    subprocess.run(cmd, shell=True)\n"),
    ("unsafe-deserialization", "import pickle\n\ndef f(data):\n    return pickle.loads(data)\n"),
    ("unsafe-deserialization", "import yaml\n\ndef f(text):\n    return yaml.load(text)\n"),
    ("sql-injection", "def f(cur, name):\n    cur.execute(f\"SELECT * FROM t WHERE n = '{name}'\")\n"),
    ("sql-injection", "def f(cur, name):\n    cur.execute('SELECT * FROM t WHERE n = ' + name)\n"),
    ("sql-injection", "def f(cur, name):\n    cur.execute('SELECT * FROM t WHERE n = %s' % name)\n"),
    ("sql-injection", "def f(cur, name):\n    cur.execute('SELECT * FROM t WHERE n = {}'.format(name))\n"),
    ("hardcoded-secret", "DB_PASSWORD = 'hunter2hunter2'\n"),
    ("hardcoded-secret", "def f():\n    connect(api_key='abcdef123456')\n"),
    ("hardcoded-secret", "CONFIG = {'secret': 'abcdef123456'}\n"),
    ("hardcoded-secret", "NOTE = 'AKIAABCDEFGHIJKLMNOP'\n"),
    ("weak-hash", "import hashlib\n\ndef f(x):\n    return hashlib.md5(x).hexdigest()\n"),
    ("insecure-random", "import random\n\ndef f():\n    token = random.randint(0, 999999)\n    return token\n"),
    ("tls-verify-disabled", "import requests\n\ndef f(u):\n    return requests.get(u, verify=False)\n"),
    ("world-writable", "import os\n\ndef f(p):\n    os.chmod(p, 0o777)\n"),
    ("assert-in-production", "def f(x):\n    assert x > 0\n    return x\n"),
    ("assert-tuple", "def f(x):\n    assert (x > 0, 'must be positive')\n"),
    ("bare-except", "def f():\n    try:\n        g()\n    except:\n        raise\n"),
    ("swallowed-exception", "def f():\n    try:\n        g()\n    except ValueError:\n        pass\n"),
    ("mutable-default", "def f(items=[]):\n    items.append(1)\n"),
    ("mutable-default", "def f(*, cache=dict()):\n    return cache\n"),
    ("open-without-with", "def f(p):\n    handle = open(p)\n    return handle.read()\n"),
    ("is-literal", "def f(x):\n    return x is 'yes'\n"),
    ("unreachable-code", "def f():\n    return 1\n    print('never')\n"),
]


@pytest.mark.parametrize("rule_id, source", VULNERABLE, ids=[f"{r}-{i}" for i, (r, _) in enumerate(VULNERABLE)])
def test_rule_fires(rule_id, source):
    assert rule_id in _rule_ids(source)


def test_every_rule_has_a_vulnerable_example():
    covered = {rule_id for rule_id, _ in VULNERABLE} | {"syntax-error"}
    assert covered == set(RULE_INFO)


CLEAN = '''"""A tidy module."""
import hashlib
import json
import os
import subprocess

import requests
import yaml

TOKEN_URL = "https://example.com/oauth/token"
SECRET_KEY = os.environ["SECRET_KEY"]
PASSWORD_FIELD = "password"


def run(args):
    return subprocess.run(["ls", *args], check=True, capture_output=True)


def load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def query(cur, name):
    cur.execute("SELECT * FROM t WHERE n = %s", (name,))
    return cur.fetchall()


def fetch(url):
    return requests.get(url, timeout=5)


def parse(text, cache=None):
    cache = cache or {}
    try:
        return json.loads(text)
    except ValueError as error:
        raise RuntimeError("bad json") from error


def check(value):
    if value is None or value is True:
        return False
    return value == "yes"
'''


def test_clean_code_has_no_findings():
    assert scan_source(CLEAN, "app.py") == []


def test_constant_arguments_are_lower_severity():
    constant = {f.rule_id: f.severity for f in scan_source("import os\nos.system('ls')\neval('1+1')\n", "a.py")}
    dynamic = {f.rule_id: f.severity for f in scan_source("import os\nos.system(cmd)\neval(code)\n", "a.py")}
    assert constant == {"shell-command": "medium", "eval-exec": "low"}
    assert dynamic == {"shell-command": "high", "eval-exec": "high"}


def test_import_aliases_are_followed():
    assert "shell-command" in _rule_ids("import subprocess as sp\nsp.run(x, shell=True)\n")
    assert "unsafe-deserialization" in _rule_ids("from pickle import loads\nloads(data)\n")
    assert "unsafe-deserialization" not in _rule_ids("import yaml\nyaml.load(t, Loader=yaml.SafeLoader)\n")


def test_secret_values_are_never_copied_into_findings():
    source = "DB_PASSWORD = 'hunter2hunter2'\nKEY = 'AKIAABCDEFGHIJKLMNOP'\n"
    findings = scan_source(source, "conf.py")
    assert len(findings) == 2
    for finding in findings:
        text = finding.model_dump_json()
        assert "hunter2" not in text and "AKIAABCDEFGHIJKLMNOP" not in text
    assert findings[0].evidence == 'DB_PASSWORD = "***"'


def test_placeholders_and_non_secret_names_are_ignored():
    source = "PASSWORD = '<your password>'\nTOKEN_URL = 'https://x.io/token'\ntoken_type = 'bearer'\nAPI_KEY = ''\n"
    assert scan_source(source, "conf.py") == []


def test_assert_is_fine_in_tests():
    source = "def test_x():\n    assert 1 + 1 == 2\n"
    assert scan_source(source, "test_x.py") == []
    assert "assert-in-production" in _rule_ids(source, "x.py")


def test_assert_tuple_is_flagged_even_in_tests():
    assert "assert-tuple" in _rule_ids("def test_x():\n    assert (False, 'boom')\n", "test_x.py")


def test_none_true_false_may_use_is():
    assert "is-literal" not in _rule_ids("def f(x):\n    return x is None or x is True or x is not False\n")


def test_with_open_is_not_flagged():
    assert "open-without-with" not in _rule_ids("def f(p):\n    with open(p) as h:\n        return h.read()\n")


def test_syntax_and_indentation_errors_are_findings():
    findings = scan_source("def f():\nreturn 1\n", "bad.py")
    assert [f.rule_id for f in findings] == ["syntax-error"]
    assert findings[0].start_line == 2 and findings[0].title.startswith("IndentationError")
    assert findings[0].severity == "high" and findings[0].kind == "bug"

    mixed = "def f():\n    if True:\n\treturn 1\n"
    assert scan_source(mixed, "tabs.py")[0].title.startswith("TabError")


def test_findings_name_their_function_and_get_stable_ids():
    source = (
        "import os\n\n\n"
        "class Runner:\n"
        "    def go(self, name):\n"
        "        os.system(name)\n"
        "\n\n"
        "def top():\n"
        "    eval(x)\n"
    )
    info = extract(source, PYTHON)
    findings = scan_source(source, "pkg/run.py", info)
    assert [(f.rule_id, f.symbol, f.start_line) for f in findings] == [
        ("shell-command", "Runner.go", 6), ("eval-exec", "top", 10),
    ]
    again = scan_source(source, "pkg/run.py", info)
    assert [f.id for f in findings] == [f.id for f in again]
    assert len({f.id for f in findings}) == 2
    assert all(f.id.startswith("F-") for f in findings)
    other_file = scan_source(source, "other/run.py", info)
    assert {f.id for f in findings}.isdisjoint(f.id for f in other_file)


def test_module_level_findings_have_no_symbol():
    findings = scan_source("import os\nos.system(cmd)\n", "a.py", extract("import os\nos.system(cmd)\n", PYTHON))
    assert findings[0].symbol is None


def test_same_line_is_reported_once_per_rule():
    findings = scan_source("API_KEY = 'AKIAABCDEFGHIJKLMNOP'\n", "a.py")
    assert [f.rule_id for f in findings] == ["hardcoded-secret"]


def test_finding_is_a_complete_record():
    finding = scan_source("def f(x):\n    return eval(x)\n", "a.py")[0]
    assert isinstance(finding, Finding)
    assert finding.kind == "vulnerability" and finding.cwe == "CWE-95" and finding.detector == "rule"
    assert finding.evidence == "return eval(x)"
    assert all([finding.title, finding.explanation, finding.scenario, finding.fix])
