"""Start a single FastAPI uvicorn worker for development."""
import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "gandalf.server:APP",
        host="0.0.0.0",
        port=6429,
        reload=True,
        reload_dirs=["gandalf"],
    )
