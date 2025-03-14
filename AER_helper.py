# from datetime import datetime, timedelta
import datetime
# from datetime import datetime
import pyodbc
from sqlalchemy import create_engine, insert, update
from sqlalchemy.engine import URL
import logging
import AER_constants


from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os
import sqlalchemy as sa
import json



class AER_helperfunction:
    def __init__(self) -> None:
        pass

    def create_sql_connection(self, server=None):
        try:

            cred_dict={}

            vault_url = os.environ['KeyVaultUrl']
            
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)

            # secretName="bigdatasql1password"
            password_selector = os.environ['servertype']
            if password_selector == 'bigdata':
                secretName = 'bigdatasql1password'
            else:
                secretName="aegsqlpassword"
            password = client.get_secret(secretName)
            cred_dict['password']=password.value

            # secretName = "bigdatasql1username"
            secretName = "aegsqlusername"
            username = client.get_secret(secretName)
            cred_dict['username']=username.value

            databasename = AER_constants.DATABASE
            cred_dict['databasename']=databasename

            drivername = os.environ['dbdriver']
            cred_dict['drivername']=drivername

            if server == 'DbServer':
                server = os.environ['dbserver']
            elif server == 'Db2Server':
                server = os.environ['dbserver2']
            else:
                server = os.environ['dbserver']
            # server = os.environ['bigdatasql1server']
            cred_dict['server']=server

            # logging.info(f"The dictionary is - {cred_dict}")

            con_string = f"DRIVER={cred_dict['drivername']};SERVER={cred_dict['server']};DATABASE={cred_dict['databasename']};UID={cred_dict['username']};PWD={cred_dict['password']}"
            logging.info(f"the connection string is - {con_string}")
            # credential = DefaultAzureCredential()

            # client = SecretClient(vault_url="https://sd-nonprod-key-vault.vault.azure.net/", credential=credential)
            # secretName="bigdatasql1password"
            # key = client.get_secret(secretName)
            # logging.info(key.value)
            
            self.connection_URL = URL.create("mssql+pyodbc", query={"odbc_connect": con_string})
            self.engine = create_engine(self.connection_URL)
            logging.info(f"create engine successful with connection url - {self.connection_URL}")
        except Exception as err:
            logging.error(f"Error when trying to connect to SQL server {err}")

        return self.engine
    
    def close_sql_connection(self):
        self.engine.dispose()

    def get_datetime_sql_string(self, date_time):
        # sql_datetime_string = date_time.strftime("%Y-%m-%d %H:%M:%S.%f %z")
        # return sql_datetime_string[:30] + ":" + sql_datetime_string[30:]
        sql_datetime_string = date_time.strftime("%Y-%m-%d %H:%M:%S")
        return sql_datetime_string
    
    def get_datetime_sql_string(self, date_time):
        # sql_datetime_string = date_time.strftime("%Y-%m-%d %H:%M:%S.%f %z")
        # return sql_datetime_string[:30] + ":" + sql_datetime_string[30:]
        sql_datetime_string = date_time.strftime("%Y-%m-%d %H:%M:%S")
        return sql_datetime_string
    
    def set_current_event_time(self, date_time):
        self.current_date_time = date_time
        
        self.end_created_date = self.current_date_time - (self.current_date_time - datetime.min) % datetime.timedelta(0, 0, 0, 0, AER_constants.default_minutes_to_process, 0, 0)
        self.begin_created_date = self.end_created_date - datetime.timedelta(0, 0, 0, 0, AER_constants.default_minutes_to_process, 0, 0)
    
    def AER_getserveripmapping(self, CIDList, server_dict):
        companyid = CIDList[0]
        database = CIDList[1]
        shardid = CIDList[2]
        for server_id, companies in server_dict.items():
            if shardid in companies:
                return server_id
        return None
    
    def AER_classify_score(self, score):
        for score_range, classification in AER_constants.score_classification_map.items():
            if score_range[0] <= score <= score_range[1]:
                return classification
        return None
    
    def DA_getautoeventclass_CID(self):
        engine = self.create_sql_connection()

        AutoEventClass_cids = self.AER_getcompany_database(engine)
        # try:
        #     json.dumps(AutoEventClass_cids)  # Try to serialize namesList to JSON
        #     logging.info("namesList is JSON serializable!")
        # except TypeError as e:
        #     logging.error(f"namesList is not JSON serializable: {e}")

        self.close_sql_connection()

        return AutoEventClass_cids
    
    def AER_getcompany_database(self, engine):

        ose_select_query = f"""
            select c.companyid as 'CompanyID', 'SafetyDirect2_shard' + convert(varchar, cc.ReportShardID) as 'DatabaseName', cc.ReportShardID as 'ShardID' from [SafetyDirect2_master].[dbo].[CompanySettings] c
                join [SafetyDirect2_master].[dbo].[Company] cc on c.COmpanyid = cc.companyid
                where c.name = 'AutoEventClassification' and c.Data like '%true%'
            """
        try:
            with engine.begin() as conn:
                result = conn.execute(sa.text(ose_select_query))
                
                # Fetch all results into a tuple of tuples
                # ComapnyList = tuple(result.fetchall())
                ComapnyList = [list(row) for row in result.fetchall()]
                logging.info(f'company details extracted with size {len(ComapnyList)}')
        except Exception as err:
            logging.error(f"Error when trying to execute ose_select_query: {err}")
        
        return ComapnyList
    
    def add_importance_score_trip_event_extra(self, engine, upsert_trip_event_extra_dataframe, databasename):
        if upsert_trip_event_extra_dataframe is None or upsert_trip_event_extra_dataframe.empty:
            logging.info(f'No data to update')
            return

        try:
            with engine.begin() as conn:
                conn.execute(sa.text("drop table if exists #upserttripeventextra;"))
                conn.close()
        except Exception as err:
            logging.error(f"Error when trying to drop #upserttripeventextra if it exists: {err}")

        upsert_trip_event_extra_dataframe.to_sql('#upserttripeventextra', con = engine, index = False, if_exists = 'replace', 
                                                 dtype = {"CompanyID": sa.types.Integer, "VehicleID": sa.types.String, 
                                                          "TripEventID": sa.types.String, "TripEventGuid": sa.types.String,
                                                          "CreatedDate": sa.types.DateTime, "importancescore": sa.types.Float})
        
        
        upsert_event_query = f"""
            MERGE {databasename}.dbo.TripEventExtra as tx
            USING (SELECT CompanyID, VehicleID, TripEventID, TripEventGuid, CreatedDate, importancescore FROM #upserttripeventextra) as utx
            ON (tx.CompanyID = utx.CompanyID AND tx.VehicleID = utx.VehicleID AND tx.TripEventID = utx.TripEventID and tx.TripEventGuid = CAST(utx.TripEventGuid as uniqueidentifier))
            WHEN MATCHED THEN
                UPDATE SET tx.ImportanceScore = utx.importancescore
            WHEN NOT MATCHED THEN 
                INSERT (CompanyID, VehicleID, TripEventID, TripEventGuid, CreatedDate, ImportanceScore)
                VALUES (utx.CompanyID, utx.VehicleID, utx.TripEventID, utx.TripEventGuid, utx.CreatedDate, utx.importancescore);
        """

        try:
            with engine.begin() as conn:
                logging.info(f'Updating scored events with size {upsert_trip_event_extra_dataframe.shape}')
                conn.execute(sa.text(upsert_event_query))
                conn.commit()
                conn.close()
        except Exception as err:
            logging.error(f"Error when trying to upsert importance score: {err}")
        
        try:
            with engine.begin() as conn:
                conn.execute(sa.text("drop table if exists #upserttripeventextra;"))
                conn.close()
                logging.info('Data Updated and temp table purged')
        except Exception as err:
            logging.error(f"Error when trying to drop #upserttripeventextra if it exists: {err}")
        
    def add_classification_type(self, engine, upsert_classification_dataframe, databasename):
        if upsert_classification_dataframe is None or upsert_classification_dataframe.empty:
            logging.info(f'No data to update')
            return

        try:
            with engine.begin() as conn:
                conn.execute(sa.text("drop table if exists #upsertclassification;"))
                conn.close()
        except Exception as err:
            logging.error(f"Error when trying to drop #upsertclassification if it exists: {err}")


        upsert_classification_dataframe.to_sql('#upsertclassification', con = engine, index = False, if_exists = 'replace', 
                                                 dtype = {"CompanyID": sa.types.Integer, "VehicleID": sa.types.String, 
                                                          "DriverID": sa.types.String, "TripEventID": sa.types.String,
                                                          "ClassificationName": sa.types.String,
                                                          "Timestamp": sa.types.DateTime, "By": sa.types.String})


        upsert_event_query = f"""
            MERGE {databasename}.dbo.EventClassification AS ec
                USING (
                    SELECT 
                        utx.CompanyID, 
                        utx.VehicleID, 
                        utx.DriverID, 
                        utx.TripEventID, 
                        ect.Type AS Classification,  -- Getting the classification value from EventClassificationType
                        utx.Timestamp, 
                        utx.[By] 
                    FROM #upsertclassification AS utx
                    INNER JOIN {databasename}.dbo.EventClassificationType AS ect
                    ON utx.ClassificationName = ect.Name  -- Joining on the classification name
                ) AS source
                ON (
                    ec.CompanyID = source.CompanyID AND 
                    ec.VehicleID = source.VehicleID AND 
                    ec.TripEventID = source.TripEventID
                )
                WHEN MATCHED THEN
                    UPDATE SET 
                        ec.Classification = source.Classification,
                        ec.Timestamp = source.Timestamp,
                        ec.[By] = source.[By]
                WHEN NOT MATCHED THEN 
                    INSERT (
                        CompanyID, 
                        VehicleID, 
                        DriverID, 
                        TripEventID, 
                        Classification, 
                        Timestamp, 
                        [By]
                    )
                    VALUES (
                        source.CompanyID, 
                        source.VehicleID, 
                        source.DriverID, 
                        source.TripEventID, 
                        source.Classification, 
                        source.Timestamp, 
                        source.[By]
                    );
        """

        try:
            with engine.begin() as conn:
                logging.info(f'Updating classification events with size {upsert_classification_dataframe.shape}')
                conn.execute(sa.text(upsert_event_query))
                conn.commit()
                conn.close()
        except Exception as err:
            logging.error(f"Error when trying to upsert importance score: {err}")
        
        try:
            with engine.begin() as conn:
                conn.execute(sa.text("drop table if exists #upsertclassification;"))
                conn.close()
                logging.info('Data Updated and temp table purged')
        except Exception as err:
            logging.error(f"Error when trying to drop #upsertclassification if it exists: {err}")