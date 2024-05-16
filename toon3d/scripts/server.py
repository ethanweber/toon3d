from http.server import SimpleHTTPRequestHandler
from pathlib import Path
import socketserver
import os
import sys
import tyro


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
        super().end_headers()


def main(
    path: Path = Path("data/processed"),
    port: int = 8000,
):
    """Start a simple HTTP server with CORS allowed for all origins.
    
    Args:
        path: The directory to serve files from.
        port: The port to serve on.
    """

    os.chdir(path)

    Handler = CORSRequestHandler
    Handler.server_version = "CustomHTTP/1.0"

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving on port {port}, CORS allowed for any origin")
        print(f"Serving files from {os.getcwd()} (relative directory)")
        httpd.serve_forever()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
