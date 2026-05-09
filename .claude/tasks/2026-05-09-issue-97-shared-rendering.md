# Issue 97 shared rendering

- Added `core.lib.render` as the headless text/usage renderer consumed by CLI and TUI.
- Added the `cost_query` service contract and registered it from `cost_budget` so presenters no longer import atom-private pricing.
- Moved tool-result diff selection to tool metadata / registered renderers and declared the edit tool's diff hint in its metadata.
