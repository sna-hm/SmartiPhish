import gym
import csv
import json
import pickle
import asyncio
import logging
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import env
import gym_register
from agent import Agent

from connection import DBConnection

logging.basicConfig(filename='logs/flask_rl_online.log', filemode='w', format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

app = Flask(__name__)
cors = CORS(app, resources={r"/moraphishdet": {"origins": "*"}, r"/moraphishup": {"origins": "*"}, r"/moraphishloaded": {"origins": "*"}})
env = gym.make('gym_register:MORAPhishDet-v0') # Make Cyberspace gym environment

with open('logs/' + 'data.csv', mode='a') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer.writerow(["date", "url", "phishing_prob", "moraphish", "g_safe", "googlesafe"])

def moraphishdet(data, url, cf, global_rank):

    env.init_dataset(data, url, external_factors=[cf, global_rank]) # input data to the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size, is_eval=True)
    done = False

    state, index = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(len(data)):
        action = agent.act(state)
        next_state, reward, done, record_id, true_reward, _ = env.step(action, state)
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if done:
            logging.warning("Rewards: {0}".format(reward))
            break

    return action, record_id

@app.route('/moraphishdet', methods=['GET', 'POST'])
def get_action():
    if not request.json or 'url' not in request.json:
        abort(400)

    content = request.json
    requested_url = content['url']

    res = requests.post('http://127.0.0.1:5001/moraphishinputs', json={"url":requested_url})
    if res.ok:
        if res.json()['status'] == 1:
            dict = json.loads(res.json()['data'])
            data = [np.array(x) for x in dict]
            url = res.json()['url']
            cf = json.loads(res.json()['cf'])
            global_rank = json.loads(res.json()['global_rank'])
            action, record_id = moraphishdet(data, url, cf, global_rank)
            #requests.post('http://127.0.0.1:5002/moraphishoffline', json={"data":res.json()['data'], "url": res.json()['url'], "cf":res.json()['cf'], "global_rank":res.json()['global_rank']})
        else:
            action = res.json()['status']
            url = [requested_url]
            record_id = 0
    else:
        action = 5
        url = [requested_url]
        record_id = 0
    #print(int(action))
    with open('logs/responses.csv', mode='a') as res_file:
        data_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([record_id, url, action])

    return jsonify({"action":int(action), "id":int(record_id), "url":str(url[0])})

@app.route('/moraphishup', methods=['GET', 'POST'])
def update():
    if not request.json or 'id' not in request.json:
        abort(400)
    content = request.json
    record_2_update = content['id']
    connection = DBConnection("smartiphish").get_connection()
    cursor = connection.cursor(buffered=True)
    sql = "SELECT * FROM model_reviews WHERE rec_id = (SELECT review_table_id FROM model_data WHERE rec_id = %s LIMIT 1) AND result = %s AND updated = %s"
    val = (record_2_update, 2, 0)
    cursor.execute(sql, val)
    if (cursor.rowcount > 0):
        sql = "UPDATE model_reviews SET status = %s WHERE rec_id = (SELECT review_table_id FROM model_data WHERE rec_id = %s LIMIT 1)"
        val = (0, record_2_update)
        cursor.execute(sql, val)
        connection.commit()
    cursor.close()
    print(str(record_2_update) + " updated")
    return jsonify({"status":"success"})

app.run(host="0.0.0.0")
