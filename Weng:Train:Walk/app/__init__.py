# Flask essentials
from flask import Flask
from flask_cors import CORS 
from flask.logging import default_handler

# System import
import os
import sys

# Logging imports
import logging
from logging import Formatter, FileHandler

from model_manager import MLModels

def create_app():

    app = Flask('flask_web')
    cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Logging config
    file_handler = FileHandler('myapp.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    app.logger.addHandler(file_handler)

    # Initial params for image path storage list and most recent id
    app.mlmodels = MLModels()

    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api/v1')

    '''
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
        return response
    '''

    return app

