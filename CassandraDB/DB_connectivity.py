import os
import pandas as pd
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from CassandraDB.app_logging.logger import Applogger
from twisted.conch.checkers import pwd


def retrieve_and_save_data():
    # Create the logger object
    logger = Applogger()

    try:
        cloud_config = {
            'secure_connect_bundle': '/Users/aakash/Downloads/secure-connect-income-database.zip'
        }

        auth_provider = PlainTextAuthProvider('pQouhmrLoLGSszZWoNYTFbpi',
                                              'Q2gL4r2wMoift2f8wlgFfO1sbR+NZMkmK11v.6v5rPMiHjrciXqDce7YDbY_+PHRiIMBJpQ1.mFbeixeB3Jpw3t+u+QT9hlfrEDh+LOUlEl3FRhiFp-28KsYCMP.Ao42')

        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        session = cluster.connect()

        row = session.execute("SELECT * FROM incomedb.adult2;")

        sql_query = "SELECT * FROM {}.{};".format('incomedb', 'adult2')

        df = pd.DataFrame(list(session.execute(sql_query)))

        folder_path = os.path.join(os.getcwd(), "LogFolder")
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, "Logfile.txt")

        # Open the log file in append mode
        with open(file_path, 'a') as file:
            file.write("Log Entry: Data successfully retrieved and saved.\n")

        file_path = os.path.join(os.getcwd(), "dataset", "full_dataset.csv")
        df.to_csv(file_path, index=False)

        with open(file_path, 'r') as file:
            file_content = file.read()
            print(file_content)
            # Process file content as needed

    except cassandra.cluster.NoHostAvailable as e:
        logger.log(file, "No Cassandra host available: " + str(e))

    except cassandra.AuthenticationFailed as e:
        logger.log(file, "Cassandra authentication failed: " + str(e))

    except cassandra.InvalidRequest as e:
        logger.log(file, "Invalid Cassandra request: " + str(e))

    except cassandra.OperationTimedOut as e:
        logger.log(file, "Cassandra operation timed out: " + str(e))

    except pd.errors.EmptyDataError as e:
        logger.log(file, "EmptyDataError occurred while creating DataFrame: " + str(e))

    except FileNotFoundError as e:
        logger.log(file, "FileNotFoundError occurred: " + str(e))

    except Exception as e:
        logger.log(file, "An unexpected error occurred: " + str(e))


# Call the function to retrieve and save the data

#retrieve_and_save_data()
