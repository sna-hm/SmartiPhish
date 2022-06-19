import gym
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
from params import Params
from datetime import datetime, timedelta

from connection import DBConnection

logging.basicConfig(filename='logs/flask_rl_offline.log', filemode='w', format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

app = Flask(__name__)
cors = CORS(app, resources={r"/moraphishoffline": {"origins": "*"}})
env = gym.make('gym_register:MORAPhishDet-v0') # Make Cyberspace gym environment

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size, is_offline=True)

params = Params()

connection = DBConnection("smartiphish").get_connection()
cursor = connection.cursor(buffered=True)

sql = "SELECT MAX(f1) FROM daily_performance;"
cursor.execute(sql)
if (cursor.rowcount > 0):
    result = cursor.fetchall()
    for row in result:
        params.max_f1 = row[0]
cursor.close()
connection.close()

def moraphishdet():
    env.init_dataset(is_offline=True) if (params.offline_mode == 0) else env.init_dataset(is_offline=True, is_adversarial=True) # input data to the environment
    done = False

    params.total_rewards = []

    if (params.offline_mode == 0):
        yesterday = datetime.now() - timedelta(1)
        yesterday = datetime.strftime(yesterday, '%Y-%m-%d')

        connection = DBConnection("smartiphish").get_connection()
        cursor = connection.cursor(buffered=True)

        sql = "SELECT id, f1 FROM daily_performance WHERE day = %s;"
        val = (str(yesterday), )
        cursor.execute(sql, val)
        if (cursor.rowcount > 0):
            result = cursor.fetchall()
            for row in result:
                params.record_id = row[0]
                params.yesterday_f1 = row[1]

        if (params.yesterday_f1 >= params.max_f1):
            params.max_f1 = params.yesterday_f1
            agent.save("./model/phishing-dqn-" + str(yesterday).replace("-", "_") + ".h5")
            logging.warning("Max. F1 updated. New F1: %.2f" % (params.max_f1))
            logging.warning("New model saved. Name: phishing-dqn-" + str(yesterday).replace("-", "_") + ".h5")

            sql = "UPDATE daily_performance SET snapshot = %s WHERE id = %s"
            val = (1, params.record_id)
            cursor.execute(sql, val)
            connection.commit()

        cursor.close()
        connection.close()

    state, index = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(params.data_count):
        action = agent.act(state)
        next_state, reward, done, record_id, true_reward, _ = env.step(action, state)
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        params.total_rewards.append(reward)

        # Start training only if certain number of samples is already saved
        if len(agent.memory) > agent.batch_size:
            agent.replay(agent.batch_size)

        if len(params.total_rewards) % 50 == 0:
            # Update target network counter every episode
            agent.target_update_counter += 1
            # If counter reaches set value, update target network with weights of main network
            if agent.target_update_counter > agent.target_update_frequency:
                agent.target_model.set_weights(agent.model.get_weights())
                agent.target_update_counter = 0

        if done:
            params.mean_reward = np.mean(params.total_rewards)
            if (params.offline_mode == 0):
                agent.save("./model/phishing-dqn.h5")
            if (params.offline_mode == 1):
                logging.warning("=====*** Adversarial Attack Training ***=====")
            logging.warning("Total reward: {}".format(np.sum(params.total_rewards[-params.data_count:])))
            logging.warning("Mean reward: %.3f" % (params.mean_reward))
            logging.warning("==========")
            break

@app.route('/moraphishoffline', methods=['GET', 'POST'])
def index():
    content = request.json
    params.data_count = int(content['data_count'])
    params.offline_mode = int(content['mode'])
    moraphishdet()

    return "done"

app.run(port='5002')
