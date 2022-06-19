import logging
import requests
from connection import DBConnection

logging.basicConfig(filename='logs/offline_train.log', filemode='a', format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

connection = DBConnection("smartiphish").get_connection()
cursor = connection.cursor(buffered=True)
data_count = 0

#update model_data and model_states
sql = "SELECT rec_id, result FROM model_reviews WHERE updated = %s"
val = (1, )
cursor.execute(sql, val)

if (cursor.rowcount > 0):
    result = cursor.fetchall()
    for row in result:
        review_tbl_id = row[0]
        record_status = row[1]

        sql = "UPDATE model_data SET result = %s WHERE review_tbl_id = %s"
        val = (record_status, review_tbl_id)
        cursor.execute(sql, val)
        connection.commit()

        sql = "SELECT rec_id FROM model_data WHERE review_tbl_id = %s"
        val = (review_tbl_id, )
        cursor.execute(sql, val)

        if (cursor.rowcount > 0):
            result = cursor.fetchall()
            model_data_tbl_id = result[0][0]

            sql = "UPDATE model_states SET status = %s, reuse_flag = %s WHERE reference_key = %s"
            val = (record_status, 1, model_data_tbl_id)
            cursor.execute(sql, val)
            connection.commit()

            sql = "UPDATE model_reviews SET updated = %s WHERE rec_id = %s"
            val = (2, review_tbl_id)
            cursor.execute(sql, val)
            connection.commit()

#truncate rl_retrain
sql = "TRUNCATE rl_retrain"
cursor.execute(sql)
connection.commit()

sql = "INSERT INTO rl_retrain (id,state,status,reference_key,reward,true_reward,reuse_flag,created_date,updated_date) SELECT * FROM model_states WHERE reuse_flag = %s;"
val = (1, )
cursor.execute(sql, val)
connection.commit()
total_ins = cursor.rowcount

sql = "SELECT COUNT(id) FROM rl_retrain WHERE status = %s"
val = (1, )
cursor.execute(sql, val)

if (cursor.rowcount > 0):
    result = cursor.fetchall()
    for row in result:
        data_count = row[0]
        limit = (data_count - (total_ins - data_count))

        if limit < 0:
            q_status = 1
        else:
            q_status = 0

        sql = "INSERT INTO rl_retrain (id,state,status,reference_key,reward,true_reward,reuse_flag,created_date,updated_date) SELECT * FROM model_states WHERE reuse_flag = %s AND status = %s AND id NOT IN (SELECT id FROM rl_retrain) ORDER BY RAND() LIMIT %s"
        val = (2, q_status, abs(limit))
        cursor.execute(sql, val)
        connection.commit()

sql = "SELECT COUNT(id) FROM rl_retrain"
cursor.execute(sql)

if (cursor.rowcount > 0):
    result = cursor.fetchall()
    for row in result:
        data_count = row[0]

cursor.close()
connection.close()

if data_count <= 1:
    pass
else:
    res = requests.post('http://127.0.0.1:5002/moraphishoffline', json={"data_count":data_count, "mode":0})
