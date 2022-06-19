import gym
import csv
import json
import pickle
import asyncio
import logging
import numpy as np
from datetime import date
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

import env
import gym_register

from connection import DBConnection
from generate_data import GenerateData
from knowledge_model import KnowledgeModel

logging.basicConfig(filename='logs/flask_rl_inputs.log', filemode='w', format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

app = Flask(__name__)
cors = CORS(app, resources={r"/moraphishinputs": {"origins": "*"}})
env = gym.make('gym_register:MORAPhishDet-v0') # Make Cyberspace gym environment
km = KnowledgeModel() # Initialize the Knowledge Model

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# loading tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

async def get_inputs(A, X, X_A, url, env, km):
    # Create task to do so:
    task1 = asyncio.ensure_future(km.get_km_prediction(A, X, X_A)) #get DL decision
    task2 = asyncio.ensure_future(env.get_google_decision(url)) #get google feedback
    task3 = asyncio.ensure_future(env.get_phishtank_decision(url)) #get phishtank feedback
    task4 = asyncio.ensure_future(env.AlexaRank(url)) #get Alexa Ranking

    # Now, when you want, you can await task finised:
    await task1, task2, task3, task4

    google_decision = task2.result()
    phishtank_decision = task3.result()
    final_decision = 1 if (google_decision or phishtank_decision) else 0

    moraphish = 1 if task1.result()[0][1] >= 0.5 else 0
    googlesafe = 1 if google_decision else 0
    today = date.today()
    with open('logs/' + 'data.csv', mode='a') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([today, url, task1.result()[0][1], moraphish, google_decision, googlesafe])

    return(task1.result(), final_decision, task4.result())

@app.route('/moraphishinputs', methods=['GET', 'POST'])
def index():
    content = request.json
    requested_url = content['url']

    gen_data = GenerateData(tokenizer)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    X, A, X_A, url = loop.run_until_complete(gen_data.generate_requested_data(requested_url))
    loop.close()

    if X == "URL GET ERROR":
        action = 3
        return jsonify({"status":int(action)})
    elif X == "DIFF URL ERROR":
        action = 4
        return jsonify({"status":int(action)})
    elif X == "NOT AN HTML ERROR":
        action = -1
        return jsonify({"status":int(action)})
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data, cf, global_rank = loop.run_until_complete(get_inputs(A, X, X_A, url[0], env, km))
        loop.close()
        logging.warning(str(url[0]) + " [successfully generated]")
        dumped_data = json.dumps(data, cls=NumpyEncoder)
        dumped_cf = json.dumps(cf, cls=NumpyEncoder)
        dumped_global_rank = json.dumps(global_rank, cls=NumpyEncoder)
        return jsonify({"status":int(1), "data":dumped_data, "url": url, "cf":dumped_cf, "global_rank":dumped_global_rank})

app.run(port='5001')
