from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from CassandraDB.app_logging.logger import Applogger
import pandas as pd
import json


class DBOperations:
    def __init__(self):
        self.log_writer = Applogger()
        self.file_object = open("CassandraDB/DBOperationLogs.txt", 'a+')
        self.client_id = 'pQouhmrLoLGSszZWoNYTFbpi'
        self.client_secret = 'Q2gL4r2wMoift2f8wlgFfO1sbR+NZMkmK11v.6v5rPMiHjrciXqDce7YDbY_+PHRiIMBJpQ1.mFbeixeB3Jpw3t+u+QT9hlfrEDh+LOUlEl3FRhiFp-28KsYCMP.Ao42'
        self.secure_connect_bundle = '/Users/aakash/Downloads/secure-connect-income-database.zip'
        self.cluster = self.data_base_connection()

    def read_schema_file(self, schema_file_path):
        try:
            with open(schema_file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            raise e


    def data_base_connection(self):
        try:
            cloud_config = {
                'secure_connect_bundle': self.secure_connect_bundle
            }
            auth_provider = PlainTextAuthProvider(username=self.client_id, password=self.client_secret)
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect()
            self.log_writer.log(self.file_object, "Connected Successfully!!!")
            return session

        except Exception as e:
            self.log_writer.log(self.file_object, f"Error while connecting to database: {e}")
            raise e

    def create_table(self, keyspace_name, table_name, column_names):
        try:
            query = f"CREATE TABLE IF NOT EXISTS {keyspace_name}.{table_name} {column_names}"
            self.session.execute(query)
            self.log_writer.log(self.file_object, f"Table '{table_name}' created successfully!!!")

        except Exception as e:
            self.log_writer.log(self.file_object, f"Error while creating table: {e}")
            raise e

    def insert_into_table(self, keyspace_name, table_name, column_names, column_values):
        try:
            columns = ', '.join(['"{column_name}"'.format(column_name=col) for col in column_names])
            values = ', '.join(['%s' for _ in column_names])
            query = f"INSERT INTO {keyspace_name}.{table_name} ({columns}) VALUES ({values})"
            self.session.execute(query, column_values)
            self.log_writer.log(self.file_object, "Inserted data into table successfully!!!")

        except Exception as e:
            self.log_writer.log(self.file_object, f"Error while inserting data into table: {e}")
            raise e

    def select_data_from_table(self, keyspace_name, table_name):
        try:
            query = f"SELECT * FROM {keyspace_name}.{table_name}"
            result_set = self.session.execute(query)
            rows = result_set.current_rows
            self.log_writer.log(self.file_object, "Selected data from table successfully!!!")
            return rows

        except Exception as e:
            self.log_writer.log(self.file_object, f"Error while selecting data from table: {e}")
            raise e

    def selecting_data_from_table_into_csv(self):
        try:
            rows = self.select_data_from_table("IncomeData", "CensusIncomeData")
            rows_list = [list(row) for row in rows]

            if len(rows_list) > 0:
                header = [desc[0] for desc in rows.column_descriptions]
                df = pd.DataFrame(rows_list, columns=header)
                df.to_csv("Training_Batch_Files/TrainingDataFromTable.csv", index=False)
                self.log_writer.log(self.file_object, "Selected data exported to CSV successfully!!!")

        except Exception as e:
            self.log_writer.log(self.file_object, f"Error while exporting data to CSV: {e}")
            raise e

