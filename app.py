# from flask import Flask, request, jsonify, make_response
# import joblib
# from flask_restx import Api, Resource, fields
# import numpy as np
# import sys
# # from sklearn.tree import DecsionTreeClassifier


# flask_app = Flask(__name__)
# app = Api(app = flask_app, 
# 		  version = "1.0", 
# 		  title = "Iris Plant identifier", 
# 		  description = "Predict the type of iris plant")

# name_space = app.namespace('prediction', description='Prediction APIs')

# model = app.model('Prediction params', 
# 				  {'sepalLength': fields.Float(required = True, 
# 				  							   description="Sepal Length", 
#     					  				 	   help="Sepal Length cannot be blank"),
# 				  'sepalWidth': fields.Float(required = True, 
# 				  							   description="Sepal Width", 
#     					  				 	   help="Sepal Width cannot be blank"),
# 				  'petalLength': fields.Float(required = True, 
# 				  							description="Petal Length", 
#     					  				 	help="Petal Length cannot be blank"),
# 				  'petalWidth': fields.Float(required = True, 
# 				  							description="Petal Width", 
#     					  				 	help="Petal Width cannot be blank")})
# # classifier = DecisionTreeClassifier()
# classifier = joblib.load('classifier.joblib')

# @name_space.route("/")
# class MainClass(Resource):

# 	def options(self):
# 		response = make_response()
# 		response.headers.add("Access-Control-Allow-Origin", "*")
# 		response.headers.add('Access-Control-Allow-Headers', "*")
# 		response.headers.add('Access-Control-Allow-Methods', "*")
# 		return response

# 	@app.expect(model)		
# 	def post(self):
# 		try: 
# 			print(request.json)
# 			formData = request.json
# 			data = [val for val in formData.values()]
# 			prediction = classifier.predict(np.array(data).reshape(1, -1))
# 			types = { 0: "Iris Setosa", 1: "Iris Versicolour ", 2: "Iris Virginica"}
# 			response = jsonify({
# 				"statusCode": 200,
# 				"status": "Prediction made",
# 				"result": "The type of iris plant is: " + types[prediction[0]]
# 				})
# 			response.headers.add('Access-Control-Allow-Origin', '*')
# 			return response
# 		except Exception as error:
# 			return jsonify({
# 				"statusCode": 500,
# 				"status": "Could not make prediction",
# 				"error": str(error)
# 			})

# if __name__ == "__main__":
#     flask_app.run(port=5000, debug=True)


from flask import Flask, request, jsonify, make_response
import joblib
from flask_restx import Api, Resource, fields
# from flask_cors import CORS
import numpy as np
import sys

flask_app = Flask(__name__)
# CORS(flask_app)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Locust predictor", 
		  description = "Predict the direction of the locust invasion")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'Latitude': fields.Float(required = True, 
				  							   description="Latitude", 
    					  				 	   help="Latitude cannot be blank"),
				  'Longitude': fields.Float(required = True, 
				  							   description="Longitude", 
    					  				 	   help="Longitude cannot be blank"),
				  'EcoLastRainStartDay': fields.Integer(required = True, 
				  							description="EcoLastRainStartDay", 
    					  				 	help="EcoLastRainStartDay cannot be blank"),
				#   'EcoLastRainStartMonth': fields.Integer(required = True, 
				#   							description="EcoLastRainStartMonth", 
    			# 		  				 	help="EcoLastRainStartMonth cannot be blank"),
				#   'EcoLastRainStartYear': fields.Integer(required = True, 
				#   							description="EcoLastRainStartYear", 
    			# 		  				 	help="EcoLastRainStartYear cannot be blank"),
				  'EcoLastRainEndDay': fields.Integer(required = True, 
				  							description="EcoLastRainEndDay", 
    					  				 	help="EcoLastRainEndDay Width cannot be blank"),
				#   'EcoLastRainEndMonth': fields.Integer(required = True, 
				#   							description="EcoLastRainEndMonth", 
    			# 		  				 	help="EcoLastRainEndMonth Width cannot be blank"),
				#   'EcoLastRainEndYear': fields.Integer(required = True, 
				#   							description="EcoLastRainEndYear", 
    			# 		  				 	help="EcoLastRainEndYear Width cannot be blank"),
				  'EcoVegDensityEst': fields.Integer(required = True, 
				  							   description="EcoVegDensityEst", 
    					  				 	   help="EcoVegDensityEst cannot be blank"),
				  'EcoVegetationState': fields.Integer(required = True, 
				  							description="EcoVegetationState", 
    					  				 	help="EcoVegetationState cannot be blank"),
				  'Infestation': fields.Integer(required = True, 
				  							description="Infestation", 
    					  				 	help="Infestation Width cannot be blank"),
				  'SwarmFlyingFrom': fields.Integer(required = True, 
				  							   description="SwarmFlyingFrom", 
    					  				 	   help="SwarmFlyingFrom cannot be blank"),
				#   'SwarmFlyingTo': fields.Integer(required = True, 
				#   							description="SwarmFlyingTo", 
    			# 		  				 	help="SwarmFlyingTo cannot be blank"),
				  'EcoSoilHumidity': fields.Integer(required = True, 
				  							description="EcoSoilHumidity", 
    					  				 	help="EcoSoilHumidity Width cannot be blank")
                                            })

classifier = joblib.load('DecisionTreeClassifier2.joblib')

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		try:
			formData = request.json
			print(formData)
			data = [val for val in formData.values()]
			prediction = classifier.predict(np.array(data).reshape(1, -1))
			print(prediction)
			score = classifier.score(np.array(data).reshape(1, -1), prediction)
			types = { 1: 'E', 2: 'S', 3: 'NE', 4: 'SE', 5: 'W', 6: 'N', 7: 'SW', 8: 'NW', 9: 'Settled'}
			flyingTo = types[prediction[0]]
			# predictionProb=classifier.predict_proba(np.array(data).reshape(1, -1)).tolist()
			# non_zero_directions=[(prob,dir+1) for dir,prob in enumerate(predictionProb) if prob>0]
			# non_zero_directions.sort(reverse=True)
			# result=non_zero_directions[0][1]
			# prob1=non_zero_directions[0][0]
			# if(len(non_zero_directions)>1):
			# 	result2=non_zero_directions[1][1]
			# 	prob2=non_zero_directions[1][0]
			# if(len(non_zero_directions)>2):
			# 	result3=non_zero_directions[2][1]
			# 	prob3=non_zero_directions[2][0]
			
			
			# print(predictionProb)
			# result2=result3=None
			# prob1=prob2=prob3=0


			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": flyingTo
			})
           
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})

if __name__ == "__main__":
    flask_app.run(port=5000, debug=True)