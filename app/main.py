from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.dependencies import init_all
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
