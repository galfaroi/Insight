from app import api

from app.controllers.yield_controller import YieldController




api.add_resource(YieldController, '/data')

