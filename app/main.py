from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    from app.dependencies import init_all

    # Enable LangSmith tracing if key is set
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        print(f"LangSmith tracing enabled → project: {settings.langchain_project}")

    print("Loading indexes and models...")
    init_all(settings)
    print("Ready.")
    yield


app = FastAPI(
    title="AI-Powered Incident Knowledge Base Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
