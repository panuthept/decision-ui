import typer
import uvicorn
from typing import Annotated
from decision_ui.server import create_app


serve_commands = typer.Typer()

@serve_commands.command("serve")
def serve_command(
    base_url: Annotated[str, typer.Argument(help="The base URL for the backend")],
    host: Annotated[str, typer.Option("--host", help="The host to serve on")] = "localhost",
    port: Annotated[int, typer.Option("--port", help="The port to serve on")] = 8000,
    timeout: Annotated[int, typer.Option("--timeout", help="The timeout for root.run() in seconds")] = 300,
    api_key: Annotated[str, typer.Option("--api_key", help="The API key for the backend")] = "EMPTY",
    mock: Annotated[bool, typer.Option("--mock", help="Enable mock mode for the backend")] = False,
):  
    # Create the FastAPI app and run it
    app = create_app(
        config={
            "backend": "vllm" if not mock else "mock",
            "base_url": base_url if base_url.endswith("/v1") else base_url + "/v1",
            "api_key": api_key,
            "timeout": timeout,
        }
    )
    uvicorn.run(app, host=host, port=port, workers=1)