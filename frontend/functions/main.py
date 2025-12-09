from firebase_functions import https_fn
from app import app

@https_fn.on_request()
def rideoptimizer(req: https_fn.Request) -> https_fn.Response:
    with app.request_context(req.environ):
        return app.full_dispatch_request()
