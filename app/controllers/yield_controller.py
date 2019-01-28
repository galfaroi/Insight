from flask_restful import Resource
from app.models.rend import YieldModel


class YieldController(Resource):

    def get(self):
        return YieldModel.all(), 200
        #return YieldModel.to_json(), 200
