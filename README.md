# WALL-E and EVE's Signal

Autonomous agent that explores a simulated environment, gathers clues, and manages limited
resources to locate EVE and decide WALL-E's next move.

> Work in progress — see `contracts/SEMANTICS.md` for the frozen contract semantics.

## Quick start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,api]"
walle run configs/sample_basic.json
pytest
```
