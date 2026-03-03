# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pydantic2linkml` is a CLI tool and library that translates [Pydantic](https://docs.pydantic.dev/) v2 models to [LinkML](https://linkml.io/) schemas. It works by introspecting Pydantic's internal `core_schema` objects rather than the higher-level model API.

## Build & Environment

This project uses [Hatch](https://hatch.pypa.io/) for environment and build management.

```bash
# Install Hatch
pip install hatch

# Run tests (in the test environment)
hatch run test:pytest tests/

# Run a single test file
hatch run test:pytest tests/test_gen_linkml.py

# Run a single test by name
hatch run test:pytest tests/test_gen_linkml.py::test_name

# Run tests with coverage
hatch run test:pytest --cov tests/

# Type checking
hatch run types:check

# Lint/format (ruff is configured in pyproject.toml)
ruff check .
ruff format .

# Spell check
codespell
```

The default hatch environment uses Python 3.9. The `test` environment adds `aind-data-schema`, `dandischema`, `pytest`, `pytest-cov`, `pytest-mock`, and `pytest-xdist`.

## CLI Usage

```bash
pydantic2linkml [OPTIONS] MODULE_NAMES...
# Example:
pydantic2linkml -o output.yml -l INFO dandischema.models
```

Options: `--output-file`/`-o` (path), `--log-level`/`-l` (default: WARNING).

## Architecture

### Core Translation Pipeline

1. **`tools.py`** — Low-level utilities for introspecting Pydantic internals:
   - `get_all_modules()` — imports modules and collects them with submodules
   - `fetch_defs()` — extracts `BaseModel` subclasses and `Enum` subclasses from modules
   - `get_field_schema()` / `get_locally_defined_fields()` — extracts resolved `pydantic_core.CoreSchema` objects for fields, distinguishing newly defined vs. overriding fields
   - `FieldSchema` (NamedTuple) — bundles a field's core schema, its resolution context, field name, `FieldInfo`, and owning model
   - `resolve_ref_schema()` — resolves `definition-ref` and `definitions` schema types to concrete schemas

2. **`gen_linkml.py`** — Main translation logic:
   - `translate_defs(module_names)` — top-level entry point; loads modules, fetches defs, runs `LinkmlGenerator`
   - `LinkmlGenerator` — single-use class; converts a collection of Pydantic models and enums into a `SchemaDefinition`. Call `generate()` once per instance.
   - `SlotGenerator` — single-use class; translates a single Pydantic `CoreSchema` into a `SlotDefinition`. Dispatches on schema `type` strings via handler methods. Handles nesting, optionality, lists, unions, literals, UUIDs, dates, etc.
   - `any_class_def` — module-level `ClassDefinition` constant for the LinkML `Any` type

3. **`cli/`** — Typer-based CLI wrapping `translate_defs`; `cli/__init__.py` defines the `app` and `main` command.

4. **`exceptions.py`** — Custom exceptions:
   - `NameCollisionError` — duplicate class/enum names across modules
   - `GeneratorReuseError` — attempting to reuse a single-use generator
   - `TranslationNotImplementedError` — schema type not yet handled
   - `SlotExtensionError` — cannot extend a base slot to match a target via slot_usage

### Key Design Patterns

- **Single-use generators**: Both `LinkmlGenerator` and `SlotGenerator` enforce one-time use via `GeneratorReuseError`. Instantiate a new object for each translation.
- **Pydantic internals**: The code directly accesses `pydantic._internal` APIs (marked with `# noinspection PyProtectedMember`). These may break on Pydantic upgrades — Pydantic is currently pinned to `~=2.7,<2.11` for this reason.
- **Field distinction**: `get_locally_defined_fields()` separates fields annotated directly on a model from those inherited, enabling correct LinkML slot vs. slot_usage generation.
- **Schema resolution**: Pydantic wraps many schemas in `definitions`/`definition-ref` indirection and function validators (`function-before`, `function-after`, etc.). `resolve_ref_schema()` and `strip_unneeded_wrapping_schema()` unwrap these before dispatch.

### Test Assets

`tests/assets/mock_module0.py` and `mock_module1.py` define Pydantic models used across test files to exercise the translator with realistic model hierarchies.
