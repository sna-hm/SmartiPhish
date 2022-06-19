import logging
import requests
from connection import DBConnection

logging.basicConfig(filename='logs/offline_train.log', filemode='a', format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

connection = DBConnection("smartiphish").get_connection()
cursor = connection.cursor(buffered=True)
data_count = 0

sql = "SELECT COUNT(id) AS p_data FROM adv_attack"
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
    res = requests.post('http://127.0.0.1:5002/moraphishoffline', json={"data_count":data_count, "mode":1})
