"""Optional richer renderers for ``agentm-terminal``.

The default text/json modes live in ``renderer.py`` and are pulled into
``cli.py`` directly. This subpackage hosts heavier frontends that
require extra dependencies — currently just ``textual`` for the
``--ui textual`` flag.
"""
