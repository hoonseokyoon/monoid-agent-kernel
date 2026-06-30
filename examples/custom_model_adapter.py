"""How to implement your own ``ModelAdapter`` — the #1 integration point.

The engine talks to any LLM through a single seam: ``next_turn(request) -> ModelTurn``.
Implement it to target your own gateway, a provider SDK, or (here) a trivial echo model.
Keep provider credentials inside the adapter; the core never sees them.

Run it::

    python examples/custom_model_adapter.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monoid_agent_kernel import (  # noqa: E402
    AgentLoop,
    AgentRunSpec,
    AgentRuntimeConfig,
)
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn  # noqa: E402


class EchoModelAdapter:
    """A ModelAdapter that finishes immediately, echoing the user's instruction.

    A real adapter would, in ``next_turn``:
      * read the turn input from ``request`` and POST it to your LLM, then
      * parse the response into a ``ModelTurn``.
    """

    # Optional capability flag the loop reads via getattr. Leave False unless your
    # adapter can accept non-text content parts.
    supports_multimodal: bool = False

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        # --- what you get (ModelRequest) ---------------------------------------------
        #   request.instruction          new user text this turn (None on a tool-only turn)
        #   request.system_prompt        the composed system prompt (regenerated each turn)
        #   request.tools                tuple[ToolSpec] visible this turn (build your
        #                                provider's tool/function schema from these)
        #   request.observations         results of tools the model called last turn
        #   request.previous_turn_handle provider handle to continue by-reference, OR
        #   request.messages             the full vendor-neutral conversation (by-value);
        #                                when set, send these and ignore the handle
        #
        # --- what you return (ModelTurn) ---------------------------------------------
        #   ModelTurn(tool_calls=(...))  ask the engine to run tools (it calls you back
        #                                with observations), OR
        #   ModelTurn(final_text="...")  settle the turn. Returning neither fails the turn.
        #   response_id / usage          optional: continuation handle and token counts.
        text = request.instruction or "(no instruction)"
        return ModelTurn(response_id="echo-1", final_text=f"echo: {text}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        spec = AgentRunSpec(workspace_root=workspace, run_root=Path(tmp) / "runs")
        config = AgentRuntimeConfig(definition_id="echo")  # no tools needed to settle

        result = AgentLoop.from_config(spec, EchoModelAdapter(), config).run_once("hello there")

        print("status     :", result.status)
        print("final_text :", result.final_text)


if __name__ == "__main__":
    main()
