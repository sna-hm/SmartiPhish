import sys
import gym
import urllib
import asyncio
import argparse
import requests
import numpy as np
import pandas as pd

from gym import spaces
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

import xml.etree.ElementTree as ET
from connection import DBConnection
from pysafebrowsing import SafeBrowsing

class Cyberspace(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.index = 0
        self.done = False
        self.actions = ["ALLOW ACCESS","STOP ACCESS","WARN USER",]
        self.google_app_key = 'AIzaSyA7PJP_ZHg6y0wvdZ2GGzBPAr4pS0vb1mc'
        self.googleSafeBrowser = SafeBrowsing(self.google_app_key)
        self.phishtank_base_url = "https://checkurl.phishtank.com/checkurl/index.php"
        self.phishtank_app_key = "0d33d3002164c826d5c93e03dedaa99b7f716843b1bef885b450bd8f2da234d9"
        self.alexarank_base_url = "http://data.alexa.com/data?cli=10&dat=s&url="
        # Action space
        self.action_space = spaces.Discrete(len(self.actions))
        # Observation space
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

    def init_dataset(self, data=None, url=None, is_offline=False, external_factors=None, is_adversarial=False):
        self.data = data
        self.url = url
        self.is_offline = is_offline
        self.is_adversarial = is_adversarial
        self.external_factors = external_factors
        self.states, self.community_feedback = self.get_states_offline() if self.is_offline else self.get_states_cf(self.data, self.url)

    def get_states_cf(self, data, url):
        l = len(data)
        processed_data = []
        cf_data = []

        res = []
        res.append(data[0][1]) #add phishing probability

        res.append(self.external_factors[0]) #add community feedback
        res.append(float(self.external_factors[1])) #add global alexa rank
        state, cf = np.array([res]), self.external_factors[0]

        for t in range(2):
            processed_data.append(state)
            cf_data.append(cf)

        return processed_data, cf_data

    def get_states_offline(self):
        offline_states = []
        offline_feedback = []
        connection = DBConnection("smartiphish").get_connection()
        cursor = connection.cursor(buffered=True)

        sql = "SELECT id, state, status FROM adv_attack ORDER BY RAND()" if self.is_adversarial else "SELECT id, state, status FROM rl_retrain ORDER BY RAND()"
        cursor.execute(sql)
        if (cursor.rowcount > 0):
            result = cursor.fetchall()
            for row in result:
                state = np.array([[float(i) for i in list(row[1].split(","))]])
                offline_states.append(state)
                offline_feedback.append([row[0], row[2]])
        cursor.close()
        return offline_states, offline_feedback

    def _next_observation(self):
        obs = self.states[self.index]
        self.index = self.index + 1
        return obs, self.index

    async def get_google_decision(self, url):
        google_decision = False
        try:
            google_decision = self.googleSafeBrowser.lookup_urls([url])[url]['malicious']
        except:
            print("Google error:", sys.exc_info()[1])
        return google_decision

    async def get_phishtank_decision(self, url):
        phishtank_decision = False
        try:
            phishtank_decision = self.PhishTank(url)
        except:
            print("PhishTank error:", sys.exc_info()[1])
        return phishtank_decision

    def PhishTank(self, url):
        in_phishtank_database = False
        phishtank_phishing = False

        try:
            r = requests.get(self.phishtank_base_url + '?url=' + url + '&app_key=' + self.phishtank_app_key + '&format=json')
            xmlDict = {}
            root = ET.fromstring(r.content)

            for child in root.iter('in_database'):
                in_phishtank_database = True if child.text == 'true' else False

            if in_phishtank_database:
                for child in root.iter('valid'):
                    phishtank_phishing = True if child.text == 'true' else False
        except:
            print("PhishTank error:", sys.exc_info()[1])
        return(phishtank_phishing)

    async def AlexaRank(self, url):
        global_rank = 0
        local_rank = 0

        try:
            r = requests.get(self.alexarank_base_url +  url)
            xmlDict = {}
            root = ET.fromstring(r.content)

            for child in root.iter("POPULARITY"): #REACH
                global_rank = child.attrib['TEXT'] #RANK

            for child in root.iter('COUNTRY'):
                local_rank = child.attrib['RANK']
        except:
            pass
        popularity = 0.0 if global_rank == 0 else 1/int(global_rank)
        return(popularity)

    def get_rewards(self, action, state):
        reward = None
        url = self.url[self.index - 1]
        DL_output = [(1 - state[0][0]), state[0][0]]
        community_decision = self.community_feedback[self.index - 1]
        record_id, user_feedback = self.get_user_feedback(url, community_decision)
        entropy = 0 if (DL_output[0] == 0 or DL_output[0] == 1) else -np.sum(DL_output * np.log2(DL_output))
        true_reward = int(entropy * 100)

        if user_feedback == 2:
            reward = int((entropy - (1 - DL_output[action]))  * 100) if (action == 0 or action == 1) else 0
        else:
            if user_feedback == action:
                reward = int(entropy * 100)
            elif action == 2:
                reward = int((entropy - DL_output[community_decision])  * 100)
            else:
                reward = int((entropy - 1) * 100)

        self.local_storage(url, action, community_decision, state, reward, entropy)
        return reward, record_id, true_reward

    def get_rewards_off(self, action, state):
        reward = None
        DL_output = [(1-state[0][0]), state[0][0]]
        record_id = self.community_feedback[self.index - 1][0]
        community_decision = self.community_feedback[self.index - 1][1]
        user_feedback = community_decision
        entropy = 0 if (DL_output[0] == 0 or DL_output[0] == 1) else -np.sum(DL_output * np.log2(DL_output))
        true_reward = int(entropy * 100)

        if user_feedback == 2:
            reward = int((entropy - (1 - DL_output[action]))  * 100) if (action == 0 or action == 1) else 0
        else:
            if user_feedback == action:
                reward = int(entropy * 100)
            elif action == 2:
                reward = int((entropy - DL_output[community_decision])  * 100)
            else:
                reward = int((entropy - 1) * 100)

        if not self.is_adversarial:
            connection = DBConnection("smartiphish").get_connection()
            cursor = connection.cursor(buffered=True)
            sql = "UPDATE model_states SET reward = %s, true_reward = %s, reuse_flag = %s WHERE id = %s"
            val = (reward, true_reward, 2, record_id)
            cursor.execute(sql, val)
            connection.commit()
            cursor.close()

        return reward, record_id, true_reward

    def local_storage(self, url, action, community_decision, state, reward, entropy):
        review_tbl_id = 0
        if community_decision == 1:
            action = community_decision
        if action != 0 or entropy > 0.8:
            review_tbl_id = self.submit_2_community(url, action)
        self.save_2_localdb(url, community_decision, review_tbl_id, state, reward)

    def step(self, action, state):
        # Execute one time step within the environment
        reward, record_id, true_reward = self.get_rewards_off(action, state) if self.is_offline else self.get_rewards(action, state)
        if self.index >= (len(self.states) - 1):
            self.done = True
        obs, _ = self._next_observation()
        return obs, reward, self.done, record_id, true_reward, {}

    def reset(self):
        self.index = 0
        self.done = False
        return self._next_observation()

    def render(self, mode='human', close=False):
        print('render')

    def submit_2_community(self, url, agent_action):
        review_tbl_id = 0
        res = requests.post('http://127.0.0.1:5010/submit', json={"key":"65c8f88191e68c69e36e49c7c8a444dd", "url":url})
        if res.ok:
            connection = DBConnection("smartiphish").get_connection()
            cursor = connection.cursor(buffered=True)

            sql = "SELECT rec_id FROM model_reviews WHERE url = %s"
            val = (str(url), )
            cursor.execute(sql, val)

            if (cursor.rowcount > 0):
                pass
            else:
                review_tbl_id = self.update_review_table(url, agent_action)

            cursor.close()
        return review_tbl_id

    def save_2_localdb(self, url, phishing_flag, review_tbl_id, state, reward):
        connection = DBConnection("smartiphish").get_connection()
        cursor = connection.cursor(buffered=True)

        sql = "SELECT rec_id, review_tbl_id, result FROM model_data WHERE url = %s"
        val = (str(url), )
        cursor.execute(sql, val)

        if (cursor.rowcount > 0):
            result = cursor.fetchall()
            record_id = result[0][0]
            db_review_tbl_id = result[0][1]
            db_result = result[0][2]

            if review_tbl_id != 0 and db_review_tbl_id == 0:
                sql = "UPDATE model_data SET review_tbl_id = %s WHERE rec_id = %s"
                val = (review_tbl_id, record_id)
                cursor.execute(sql, val)
                connection.commit()

                sql = "UPDATE model_states SET reuse_flag = %s WHERE reference_key = %s AND reuse_flag > %s"
                val = (0, record_id, 0)
                cursor.execute(sql, val)
                connection.commit()

            if db_result != phishing_flag and db_result != 1:
                sql = "UPDATE model_data SET result = %s WHERE rec_id = %s"
                val = (phishing_flag, record_id)
                cursor.execute(sql, val)
                connection.commit()

        else:
            sql = "INSERT INTO model_data(url, result, review_tbl_id) VALUES(%s, %s, %s)"
            val = (str(url), phishing_flag, review_tbl_id)
            cursor.execute(sql, val)
            connection.commit()
            last_record_id = cursor.lastrowid

            #save states to model_states table
            sql = "INSERT INTO model_states(state, status, reference_key, reward) VALUES(%s, %s, %s, %s)"
            val = (','.join(map(str, state[0].tolist())), phishing_flag, last_record_id, reward)
            cursor.execute(sql, val)
            connection.commit()
        cursor.close()

    def update_review_table(self, url, agent_action):
        connection = DBConnection("smartiphish").get_connection()
        cursor = connection.cursor(buffered=True)
        sql = "INSERT INTO model_reviews(url, result) VALUES(%s, %s)"
        val = (str(url), agent_action)
        cursor.execute(sql, val)
        connection.commit()
        _last_record_id = cursor.lastrowid
        cursor.close()
        return _last_record_id

    def get_user_feedback(self, url, community_decision):
        feedback = community_decision
        record_id = 0
        connection = DBConnection("smartiphish").get_connection()
        cursor = connection.cursor(buffered=True)

        sql = "SELECT rec_id, result FROM model_reviews WHERE url = %s"
        val = (str(url), )
        cursor.execute(sql, val)

        if (cursor.rowcount > 0):
            result = cursor.fetchall()
            record_id = result[0][0]
            feedback = result[0][1]
            if feedback == 0 and community_decision == 1:
                sql = "INSERT INTO model_log(type,related_table, related_id, message, flag) VALUES(%s, %s, %s)"
                val = ("1", "model_reviews", record_id, "PhishRepo reported a legitimate, however, community decision tells phishing ", "1") # flag=1 means highest priority
                cursor.execute(sql, val)
                connection.commit()
        cursor.close()
        return record_id, feedback
