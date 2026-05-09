# Changelog

## Unreleased

- Changed Claude Code compatibility loading from the old flat atom modules
  `contrib.extensions.cc.cc_agents`, `contrib.extensions.cc.cc_commands`, and
  `contrib.extensions.cc.cc_plugins` to the explicit package mount
  `contrib.extensions.cc`. Existing scenario manifests should replace the
  three old dotted paths with one `--extension contrib.extensions.cc` or
  manifest entry for `contrib.extensions.cc`.
