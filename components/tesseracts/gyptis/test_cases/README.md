# Test cases

Each `*.json` file in this directory is a regression test run by `make test`,
which invokes `tesseract run <tesseract_name> test @<file>` to verify it. A test
case specifies an endpoint, its input `payload`, and either `expected_outputs`
or an `expected_exception` (with optional `atol`/`rtol` for numeric outputs).

```json
{
    "endpoint": "apply",
    "payload": {"inputs": {"x": 0.0, "y": 0.0}},
    "expected_outputs": {"result": 1.0},
    "atol": 1e-8,
    "rtol": 1e-5
}
```

Array outputs are compared with `atol`/`rtol`. To capture the expected outputs
for a new case automatically, put an input `payload` in a JSON file and let
`make gen-tests` run the (already built) component and record its output:

```bash
# payload.json:  {"inputs": {"x": 0.0, "y": 0.0}}
make gen-tests <tesseract_name> FILE=payload.json
```

This writes a ready-to-run case here. Review it before committing — tolerances
and non-deterministic outputs may need hand-editing. To do it by hand instead,
run the endpoint once and copy its output into `expected_outputs`:

```bash
tesseract run <tesseract_name> apply '{"inputs": {"x": 0.0, "y": 0.0}}'
```

The shipped `example.json` exercises the empty base schema and passes as-is;
replace it once your `InputSchema`/`OutputSchema` have real fields.
