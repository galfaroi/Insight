import numpy as np
import GPy
import pickle
from flask_restful import Resource, reqparse
from flask_restful import Api
import json
normalization_data = {'std': mean, 'std': std, 'mean_semana': mean_semana, 'std_semana': std_semana, 'mean_dia': mean_dia, \
                      'std_dia': std_dia, 'mean_sem_tra': mean_sem_tra, 'std_sem_tra': std_sem_tra}

class Forecast(Resource):
# Model creation, without initialization:
    def get(self):
        m_= pickle.load( open( "modelGPy.p", "rb" ) )
        with open('/home/german/Desktop/insight_project/norm.json', 'rb') as f:
            norm  = json.loads(open('/home/german/Desktop/insight_project/norm.json').read())
        semana = (semana - norm['mean_semana'])/norm['std_semana']
        dia = (dia - norm['mean_dia'])/norm['std_dia']
        sem_trans = (sem_trans- norm['mean_sem_tra'])/norm['std_sem_tra']
        x = np.array([semana, dia, sem_trans])
        y_predict_gpy_ = m_.predict(x)
        quantiles_ = m_.predict_quantiles(x)
        conf_down =quantiles[0]
        conf_up = quantiles[1]
        y_predict_gpy = y_predict_gpy[0]



        output = {'prediction': y_predict_gpy , 'confidence_up': conf_up, 'confidence_down': conf_down}

        return output

    def put(self):
        parser.add_argument('semana', help = 'This field cannot be blank', required = True)
	    parser.add_argument('dia_transplante', help = 'This field cannot be blank', required = True)
	    parser.add_argument('semana_trans', help = 'This field cannot be blank', required = True)
	    parser = reqparse.RequestParser()
        semana = [config['semana']
        dia =  config['dia_transplante']
        sem_trans = config['semana_trans']
        #x = np.array([config['semana'], config['dia_transplante'],config['semana_trans']])
