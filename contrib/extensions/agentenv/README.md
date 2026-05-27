# agentm-agentenv

AgentM ⟷ ARL [agent-env](https://github.com/Lincyaw/agent-env) sandbox
integration, packaged as an **installable plugin** (entry points) rather than
a source-checkout contrib file.

Ships two things, discovered **by name** once installed (no source checkout,
no `AGENTM_PROJECT_ROOT`, no path):

| Kind | Name | Entry-point group |
|------|------|-------------------|
| atom | `operations_agent_env` | `agentm.atoms` |
| scenario | `agent_env_repo` | `agentm.scenarios` |

```bash
pip install "agentm[agent-env]"      # or: uv sync --extra agent-env
agentm --scenario agent_env_repo "…" # resolves from the installed wheel
```

`agent_env_repo` = the sandboxed Operations bundle (`bash` + file I/O against a
pod's `/workspace`) with `sync_cwd: true`: the pod is seeded from the host
cwd's `git HEAD` before the run and the agent's diff is applied back on
shutdown — so a dispatcher (e.g. workbuddy) can clone with its own creds, hand
AgentM the work tree, and collect the in-pod edits for a PR, **without any VCS
credential entering the pod**.

## Authoring your own plugin (the consistent convention)

```toml
[project.entry-points."agentm.atoms"]
my_atom = "my_pkg.my_atom"            # module exposing MANIFEST + install

[project.entry-points."agentm.scenarios"]
my_scenario = "my_pkg.my_scenario"    # package whose dir holds manifest.yaml
                                      # (dir name == scenario name)
```

`pip install` your distribution and AgentM finds both by name.
