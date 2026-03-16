# Task: Vault Parser Module

**Date**: 2026-03-16
**Status**: PENDING
**Plan**: [plan](../plans/2026-03-16-memory-vault.md)
**Design**: [design](../designs/memory-vault.md)
**Assignee**: tdd

## Objective

Implement `src/agentm/tools/vault/parser.py` — pure functions for parsing and serializing Markdown files with YAML frontmatter, extracting `[[wikilink]]` references, and parsing/manipulating sections.

## Inputs

- [Design doc § File Format, § Frontmatter Fields, § Link Syntax, § Edit Operations](../designs/memory-vault.md)

## Outputs

- `src/agentm/tools/vault/parser.py` (new)
- `tests/unit/test_vault_parser.py` (new)

## Implementation Details

### Functions to implement

1. **`parse_note(content: str) -> tuple[dict, str]`**
   - Split `---` delimited YAML frontmatter from body
   - Return `(frontmatter_dict, body_str)`
   - Preserve unknown fields in frontmatter (extensible)
   - Handle missing frontmatter gracefully (return empty dict)

2. **`serialize_note(frontmatter: dict, body: str) -> str`**
   - Render frontmatter dict as YAML between `---` fences + body
   - Maintain consistent field ordering where possible

3. **`extract_wikilinks(text: str) -> list[str]`**
   - Regex: `\[\[([^\]]+)\]\]`
   - Extract from both frontmatter string values and body
   - Return deduplicated list of paths (without `.md` extension)

4. **`extract_title(body: str) -> str`**
   - Extract first `# ` heading from body
   - Return empty string if no h1 found

5. **`find_section(body: str, heading: str) -> tuple[int, int] | None`**
   - Find the start and end char offsets of a section
   - Section ends at next heading of same or higher level, or end of file
   - `heading` must match exactly including `##` level

6. **`replace_section(body: str, heading: str, new_content: str) -> str`**
   - Replace content under heading (preserve heading line itself)
   - Use `find_section` internally

7. **`append_to_section(body: str, heading: str, content: str) -> str`**
   - Append content before next heading at same/higher level
   - Use `find_section` internally

8. **`replace_string(body: str, old: str, new: str) -> str`**
   - Exact string replacement (first occurrence)
   - Raise ValueError if `old` not found
   - Raise ValueError if `old` appears multiple times (ambiguous)

## Acceptance Conditions

- [ ] Frontmatter round-trip: parse then serialize preserves all fields
- [ ] Unknown frontmatter fields preserved (extensibility)
- [ ] Missing frontmatter returns empty dict, full body
- [ ] Wikilink extraction finds links in body and nested frontmatter strings
- [ ] Section replacement handles edge cases: last section, empty section, nested headings
- [ ] replace_string raises on not-found and on ambiguous (multiple matches)
- [ ] Title extraction works with and without h1
- [ ] All functions are pure (no side effects, no mutation)
- [ ] No file I/O in this module

## Notes

- Use `yaml.safe_load` / `yaml.safe_dump` from PyYAML (already a dependency)
- Keep module under 200 lines — these are pure string operations
- Consider edge case: frontmatter containing `---` within a multi-line YAML value
