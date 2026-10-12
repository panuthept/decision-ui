import typer
import uvicorn
from typing import Annotated
from decision_ui.server import create_app


serve_commands = typer.Typer()

@serve_commands.command("serve")
def serve_command(
    host: Annotated[str, typer.Option("--host", help="The host to serve on")] = "localhost",
    port: Annotated[int, typer.Option("--port", help="The port to serve on")] = 8000,
    timeout: Annotated[int, typer.Option("--timeout", help="The timeout for root.run() in seconds")] = 300,
    backend: Annotated[str, typer.Option("--backend", help="The backend to use for the RØØT server")] = "vllm",
    base_url: Annotated[str, typer.Option("--base-url", help="The base URL for the backend")] = "http://localhost:8000",
    api_key: Annotated[str, typer.Option("--api-key", help="The API key for the backend")] = "EMPTY",
    mock: Annotated[bool, typer.Option("--mock", help="Enable mock mode for the backend")] = False,
):  
    # Create the FastAPI app and run it
    app = create_app(
        config={
            "backend": backend if not mock else "mock",
            "base_url": base_url,
            "api_key": api_key,
            "timeout": timeout,
        }
    )
    uvicorn.run(app, host=host, port=port, workers=1)