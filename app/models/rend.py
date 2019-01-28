from app import db
from sqlalchemy.orm import aliased


class YieldModel(db.Model):

    __bind_key__ = 'yield'
    __tablename__ = 'produccion'

    id = db.Column(db.Integer, primary_key=True)
    temporada = db.Column(db.Integer, nullable=False)
    semana = db.Column(db.Integer, nullable=False)
    malla  = db.Column(db.String, nullable=False)
    rendimiento = db.Column(db.Float, nullable=False)
    dia_transplante = db.Column(db.Integer, nullable=False)
    semana_trans = db.Column(db.Integer, nullable=False)


    def __str__(self):
        return "<YieldModel(id=%s,temporada=%s,semana=%s,malla=%s,rendimiento=%s,dia_transplante=%s,semana_trans=%s)>" % \
               (self.id, self.temporada, self.semana, self.malla, self.rendimiento, self.dia_transplante, self.semana_trans)

    def to_json(item):
        return {
            'id': item.id,
            'temporada': item.temporada,
            'semana': item.semana,
            'malla': item.malla,
            'rendimiento': item.rendimiento,
            'dia_transplante': item.dia_transplante,
	    'semana_trans': item.semana_trans

        }

    @classmethod
    def all(cls):

        yield_alias = aliased(YieldModel, name='yield_alias')

        test = YieldModel.query.all()

        return list(map(lambda x: cls.to_json(x), test))
