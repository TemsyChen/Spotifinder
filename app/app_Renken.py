from .db import DB, Artist

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app import db, ml, viz

def create_app():

    app = FastAPI(
        title='Spotifinder',
        description="Find the next best song for you based on your personal tastes and preferences",
        docs_url='/',
    )

    app.include_router(ml.router, tags=['Machine Learning'])
    app.include_router(viz.router, tags=['Visualization'])
    app.include_router(db.router, tags=['Database'])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    #initialize database
    DB.init_app(app)

    # @app.route("/db")

app=create_app()

if __name__ == '__main__':
    uvicorn.run(app)

