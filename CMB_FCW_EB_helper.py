import pandas as pd
from collections import defaultdict
import datetime
import sqlalchemy as sa
import numpy as np
import datetime
################################################################################################################################################################
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier


import pyodbc
from sqlalchemy import create_engine, insert, update
from sqlalchemy.engine import URL

import logging
import os
import joblib
import AER_helper
import AER_constants
import datetime
####################################################################################################################################################

# class CMB_FCW_EB_Helper:
#     def __init__(self) -> None:
#         pass


#     def create_sql_connection(self):
#         try:
#             con_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.11.1.8;DATABASE=SafetyDirect2_shard63;UID=sdWebSite;PWD=Truck1ngD@ta!"
#             logging.info(f"the connection string is - {con_string}")
#             # credential = DefaultAzureCredential()

#             # client = SecretClient(vault_url="https://sd-nonprod-key-vault.vault.azure.net/", credential=credential)
#             # secretName="bigdatasql1password"
#             # key = client.get_secret(secretName)
#             # logging.info(key.value)
            
#             self.connection_URL = URL.create("mssql+pyodbc", query={"odbc_connect": con_string})
#             self.engine = create_engine(self.connection_URL, future=True)
#             logging.info(f"create engine successful with connection url - {self.connection_URL}")
#         except Exception as err:
#             raise Exception(f"Error when trying to connect to SQL server {err}")

#         return self.engine
    
#     def close_sql_connection(self):
#         self.engine.dispose()


class CMB_FCW_EB_Classification:

    def __init__(self):
        self.CMBFCWEB_helper = CMB_FCW_EB()
        self.AER_helper = AER_helper.AER_helperfunction()


    def CMBFCWEB_get_data(self, CIDList, server):
        engine = self.AER_helper.create_sql_connection(server)

        companyid = CIDList[0]
        database = CIDList[1]
        shardid = CIDList[2]

        ECS_event_dataframe = self.CMBFCWEB_helper.create_CMBFCWEB_event_dataframe(engine, database, companyid)

        self.AER_helper.close_sql_connection()

        return ECS_event_dataframe
    
    def CMBFCWEB_update_data(self, upsert_trip_event_extra_dataframe,upsert_classification_dataframe, CIDList, server):
        # logging.info(f"func_CMB_FCW_EB updating data with data size {upsert_trip_event_extra_dataframe.shape}")
        engine = self.AER_helper.create_sql_connection(server)

        companyid = CIDList[0]
        database = CIDList[1]
        shardid = CIDList[2]

        logging.info(f'CIDlist is {CIDList} and database is {database}')

        self.AER_helper.add_importance_score_trip_event_extra(engine, upsert_trip_event_extra_dataframe, database)

        self.AER_helper.add_classification_type(engine, upsert_classification_dataframe, database)

        self.AER_helper.close_sql_connection()
        logging.info(f"func_CMB_FCW_EB for companyid {companyid} Data updated")

    def DA_getautoeventclass_CID(self):
        engine = self.AER_helper.create_sql_connection()

        AutoEventClass_cids = self.AER_helper.AER_getcompany_database(engine)

        self.AER_helper.close_sql_connection()

        return AutoEventClass_cids


class CMB_FCW_EB:

    def set_initial_parameters(self):
        
        self.minutes_before_process = AER_constants.minutes_before_process
        self.current_date_time = datetime.datetime.utcnow()
        self.current_date_time = self.current_date_time

        if(self.minutes_before_process is None):
            self.minutes_before_process = AER_constants.default_minutes_to_process

        # -- These variables establish the TripEvent CreatedDate range that will be processed.
	    # -- For example, if the "@minutesBeforeProcess" value is 30 (minutes), the dates will be the two most recent multiples of 30 minutes.
	    # -- Therefore, if this function is executed at the datetime of "2023-01-01 08:02:00",
	    # --		then the "beginCreatedDate" value will be "2023-01-01 07:30:00" and the "endCreatedDate" value will be "2023-01-01 08:00:00".
	    # -- This establishes well-defined CreatedDate boundaries for processing data.
	    # -- In this situation, it is assumed that the job that executes this stored procedure will run every 30 minutes, or whatever the value of "@minutesBeforeProcess" is.

        minutes_before_event = (int((self.current_date_time - datetime.datetime.min).total_seconds() / 60 / self.minutes_before_process) * self.minutes_before_process)

        self.end_created_date = datetime.datetime.min + datetime.timedelta(minutes=minutes_before_event)
        self.begin_created_date = datetime.datetime.min + datetime.timedelta(minutes=minutes_before_event - self.minutes_before_process)

    def create_CMBFCWEB_event_dataframe(self, engine, database, companyid):
        self.set_initial_parameters()

        begin_created_date_string = self.begin_created_date.strftime("%Y-%m-%d %H:%M:%S")
        end_created_date_string = self.end_created_date.strftime("%Y-%m-%d %H:%M:%S")
        

        # begin_created_date_string = '2024-09-27 16:00:00'
        # end_created_date_string = '2024-09-27 20:00:00'
        logging.info(f'time selected is from {begin_created_date_string} to {end_created_date_string}')
        logging.info("func_CMB_FCW_EB inside creating dataframe")

       
        DA_select_query = f"""
            select
                t.CompanyID, 
                t.TripID,
                t.CreatedDate,
                t.TripEventID,
                t.TripEventGuid,
                t.VehicleID,
                t.DriverID,
                t.SevereEvent,
                t.Speed as 'teSpeed',
                p.speed as 'ppeSpeed',
                t.BrakingForce as 'teBrakingForce',
                p.BrakingForce as 'ppeBrakingForce',
                t.TurningForce as 'teTurningForce',
                p.TurningForce as 'ppeTurningForce',
                t.Curvature as 'tuCurvature',
                p.Curvature as 'ppeCurvature',
                t.BrakePercentage as 'teBrakePercentage',
                p.BrakePercentage as 'ppeBrakePercentage',
                t.Timestamp as 'EventTimeStamp',
                p.Timestamp as 'ppeTimeStamp',
                CONVERT(varchar, DATEDIFF(second, t.Timestamp, p.Timestamp)) + '.' + CONVERT(varchar, p.SequenceID) as 'TimeSegment',
                DATEDIFF(second, t.Timestamp, p.Timestamp) + ((p.SequenceID - 1) * 0.25) as 'SecTimeSegment',
                0 as isSevere
                from {database}.dbo.TripEvent t 
                JOIN {database}.dbo.PrePostEvent p 
                    on p.CompanyID = t.CompanyID
                    AND p.VehicleID = t.VehicleID
                WHERE t.EventID in (32, 4096, 1024)
                AND t.CompanyID = {companyid}
                AND t.CreatedDate < '{end_created_date_string}'
                AND t.CreatedDate >= '{begin_created_date_string}'
                AND p.Timestamp > DATEADD(second, -11, t.timestamp)
                AND p.Timestamp < DATEADD(second, 11, t.timestamp)
        """
        
        # try:
        with engine.begin() as conn:
            CMBFCWEB_event_dataframe = pd.read_sql_query(sa.text(DA_select_query), conn)
        # except Exception as err:
        #     raise Exception(f"Error when trying to execute DA_select_query: {err}")

        if CMBFCWEB_event_dataframe is None or CMBFCWEB_event_dataframe.empty:
            logging.info(f'func_CMB_FCW_EB no data from {begin_created_date_string} to {end_created_date_string}')
            return None
        else:

            logging.info(f"func_CMB_FCW_EB Data retrieved with size as {CMBFCWEB_event_dataframe.shape}")
            return CMBFCWEB_event_dataframe
        

    
class data_preperation:

    def __init__(self):
        self.AER_helper = AER_helper.AER_helperfunction()

    def RatioOfSpeedChange(self,df):
        
        # Split the DataFrame based on SecTimeSegment and calculate first and last PPEspeed in the same line
        df_before_zero = df[df['SecTimeSegment'] < 0].groupby(['DriverID', 'TripID', 'TripEventID'])['ppeSpeed'].agg(First_PPEspeed='first', Last_PPEspeed='last').reset_index()
        df_after_zero = df[df['SecTimeSegment'] >= 0].groupby(['DriverID', 'TripID', 'TripEventID'])['ppeSpeed'].agg(First_PPEspeed='first', Last_PPEspeed='last').reset_index()
        
        # Calculate DeltaSpeedBeforeZero and DeltaSpeedAfterZero
        df_before_zero['DeltaSpeedBeforeZero'] = df_before_zero['First_PPEspeed'] - df_before_zero['Last_PPEspeed']
        df_after_zero['DeltaSpeedAfterZero'] = df_after_zero['First_PPEspeed'] - df_after_zero['Last_PPEspeed']
        
        # Perform the left join
        left_join = pd.merge(df_before_zero, 
                             df_after_zero, 
                             on=['DriverID', 'TripID', 'TripEventID'], 
                             how='left')
        
        # Perform the right join
        right_join = pd.merge(df_before_zero, 
                              df_after_zero, 
                              on=['DriverID', 'TripID', 'TripEventID'], 
                              how='right')
        
        # Perform the inner join
        inner_join = pd.merge(df_before_zero, 
                              df_after_zero, 
                              on=['DriverID', 'TripID', 'TripEventID'], 
                              how='inner')
        
        # Union all results (concatenate and drop duplicates)
        union_all = pd.concat([left_join, right_join, inner_join]).drop_duplicates().reset_index(drop=True)
        
        union_all['DeltaSpeedRatio'] = union_all['DeltaSpeedBeforeZero']/(0.001+ union_all['DeltaSpeedAfterZero'])
        
        return union_all


    def process_dataframe(self, df):
        # 1. Split the data based on AbsValBrakingForce > 0.1
        df_split_positive = df[df['AbsValBrakingForce'] > 0.1]
        df_split_negative = df[df['AbsValBrakingForce'] <= 0.1]
        
        # 2. For positive split, group by TripEventID and get first and last values of SecTimeSegment
        df_positive_grouped = df_split_positive.groupby('TripEventID')['SecTimeSegment'].agg(First_SecTimeSegment='first', Last_SecTimeSegment='last').reset_index()
        
        # 3. For negative split, hardcode values for First_SecTimeSegment and Last_SecTimeSegment
        df_split_negative['First_SecTimeSegment'] = 10
        df_split_negative['Last_SecTimeSegment'] = -10
        df_negative_grouped = df_split_negative[['TripEventID', 'First_SecTimeSegment', 'Last_SecTimeSegment']].drop_duplicates()
        
        # 4. Union the results of steps 2 and 3
        df_union = pd.concat([df_positive_grouped, df_negative_grouped])
        
        # 5. Group by TripEventID and get the min of First_SecTimeSegment and max of Last_SecTimeSegment
        result = df_union.groupby('TripEventID').agg(Min_First_SecTimeSegment=('First_SecTimeSegment', 'min'), 
                                                     Max_Last_SecTimeSegment=('Last_SecTimeSegment', 'max')).reset_index()
        
        # 6. Return the result
        return result

    def brakeTimesteps(self, df):
        df_group = df.groupby('TripEventID')['AbsValBrakingForce'].agg(Count='count').reset_index()
        df_split_positive = df[df['AbsValBrakingForce'] > 0.1]
        df_positive_grouped = df_split_positive.groupby('TripEventID')['SecTimeSegment'].agg(count='count').reset_index()
        df_union = pd.concat([df_positive_grouped, df_group])
        df_union_groups = df_union.groupby('TripEventID')['Count'].agg(AfterZeroBrakeOnCount='max').reset_index()
        return df_union_groups
    
    def TurningIrregularity(self,df):
        df['TFCenterDeviation'] = np.where((df.DriverID == df.DriverID.shift(1)) &  (df.TripEventID == df.TripEventID.shift(1)) & (df.DriverID.shift(1) == df.DriverID.shift(2)) &  (df.TripEventID.shift(1) == df.TripEventID.shift(2)), 
                                                     abs(df.ppeTurningForce.shift(1) - 0.5* (df.ppeTurningForce + df.ppeTurningForce.shift(2))), np.nan)
        
        df_groups = df.groupby('TripEventID')['TFCenterDeviation'].agg(Sum_TFCenterDeviation='sum', Max_TFCenterDeviation = 'max').reset_index()
        return df_groups
    
    def BrakingIrregularity(self,df):
        df['BFCenterDeviation'] = np.where((df.DriverID == df.DriverID.shift(1)) &  (df.TripEventID == df.TripEventID.shift(1)) & (df.DriverID.shift(1) == df.DriverID.shift(2)) &  (df.TripEventID.shift(1) == df.TripEventID.shift(2)), 
                                                     abs(df.ppeBrakingForce.shift(1) - 0.5* (df.ppeBrakingForce + df.ppeBrakingForce.shift(2))), np.nan)
        
        df_groups = df.groupby('TripEventID')['BFCenterDeviation'].agg(Sum_BFCenterDeviation='sum', Max_BFCenterDeviation = 'max').reset_index()
        return df_groups
    
    def sppedcountafter0(self,df):
        df['SpeedChange'] = np.where((df.DriverID == df.DriverID.shift(1)) &  (df.TripEventID == df.TripEventID.shift(1)), 
                                                 abs(df.ppeSpeed - df.ppeSpeed.shift(1)), np.nan)
        df = df[(df['SecTimeSegment'] >= 0) & (df['SpeedChange'] > 0)]
        df_groups = df.groupby('TripEventID')['SpeedChange'].agg(SIC='count').reset_index()
        
        df = pd.merge(df, 
                    df_groups, 
                    on=['TripEventID'], 
                    how='left')
        
        return df
    
    def feature_generation(self,df):

        # logging.info('checkpoint 1')
        df.reset_index(drop=True, inplace=True)
        df.dropna(subset=['TripID'], inplace=True)
        df = df.sort_values(by = ['DriverID', 'TripID', 'TripEventID', 'SecTimeSegment'], ascending = [True, True, True, True])

        df['SecTimeSegment_Sec'] = df['SecTimeSegment'].astype(int)
        df['AbsValTurningForce'] = abs(df['ppeTurningForce'])
        df['AbsValBrakingForce'] = abs(df['ppeBrakingForce'])

        df['PrevTF'] = np.where((df.TripID == df.TripID.shift(1)) &(df.DriverID == df.DriverID.shift(1)) &  (df.TripEventID == df.TripEventID.shift(1)), 
                                                     df.ppeTurningForce.shift(1), np.nan)

        df['CenterDifferenceDeviation'] = np.where((df.DriverID == df.DriverID.shift(1)) &  (df.TripEventID == df.TripEventID.shift(1)) & (df.DriverID == df.DriverID.shift(-1)) &  (df.TripEventID == df.TripEventID.shift(-1)), 
                                                     abs(df.ppeSpeed - 0.5* (df.ppeSpeed.shift(1) + df.ppeSpeed.shift(-1))), np.nan)


        df['PrevPrevTF'] = np.where((df.TripID == df.TripID.shift(2)) &(df.DriverID == df.DriverID.shift(2)) &  (df.TripEventID == df.TripEventID.shift(2)), 
                                                     df.ppeTurningForce.shift(2), np.nan)
        
        df_copy = df.copy()
# 
        # logging.info('checkpoint 2')
        df['AbsVal_TF_Moving_Delta'] = np.where((df.DriverID == df.DriverID.shift(1)) &  (df.TripEventID == df.TripEventID.shift(1)), 
                                                     abs(df.ppeTurningForce - df.ppeTurningForce.shift(1)), np.nan)

        df['AbsDiffBrakingForce'] = np.where((df.DriverID == df.DriverID.shift(1)) &  (df.TripEventID == df.TripEventID.shift(1)), 
                                                     abs(df.ppeBrakingForce - df.ppeBrakingForce.shift(1)), np.nan)
        
        df1 = (df.groupby(['DriverID','TripEventID'])['ppeSpeed']
                 .agg(SpeedChange=lambda x: abs(x.iat[-1] - x.iat[0]),
                      EndSpeedOverStartSpeed=lambda x: x.iat[-1] / x.iat[0] if x.iat[0] != 0 else None)
                 .reset_index())

        df = df.merge(df1, on=['DriverID', 'TripEventID'], how='left')

        df = self.sppedcountafter0(df)
        
        left_join = pd.merge(df , self.RatioOfSpeedChange(df_copy),
                             on=['TripEventID'], 
                             how='left')

        inner_join = pd.merge(df , self.RatioOfSpeedChange(df_copy),
                             on=['TripEventID'], 
                             how='inner')

        union_all = pd.concat([left_join, inner_join]).drop_duplicates().reset_index(drop=True)
        # logging.info('checkpoint 3')
        #######################
        left_join = pd.merge(union_all , self.process_dataframe(df_copy),
                             on=['TripEventID'], 
                             how='left')

        inner_join = pd.merge(union_all , self.process_dataframe(df_copy),
                             on=['TripEventID'], 
                             how='inner')

        union_all = pd.concat([left_join, inner_join]).drop_duplicates().reset_index(drop=True)

        # logging.info('checkpoint 4')
        #################
        left_join = pd.merge(union_all , self.brakeTimesteps(df_copy),
                             on=['TripEventID'], 
                             how='left')

        inner_join = pd.merge(union_all , self.brakeTimesteps(df_copy),
                             on=['TripEventID'], 
                             how='inner')

        union_all = pd.concat([left_join, inner_join]).drop_duplicates().reset_index(drop=True)


        # logging.info('checkpoint 5')
        ###########
        df_copy_group1 = df_copy.groupby('TripEventID').agg(Max_AbsValTurningForce=('AbsValTurningForce', 'max'),
                                                            Max_AbsValBrakingForce=('AbsValBrakingForce', 'max'),
                                                            Lastest_PPESpeed=('ppeSpeed', 'last')).reset_index()

        union_all = pd.merge(union_all, df_copy_group1,
                             on=['TripEventID'],
                             how='inner')
        # logging.info('checkpoint 6')
        ##############


        df_copy_group1 = df_copy.groupby('TripEventID')['CenterDifferenceDeviation'].agg(Sum_CenterDifferenceDeviation = 'sum', Max_CenterDifferenceDeviation = 'max').reset_index()

        union_all = pd.merge(union_all, df_copy_group1,
                             on=['TripEventID'],
                             how='inner')


        ##########

        union_all = pd.merge(union_all, self.TurningIrregularity(df_copy),
                             on=['TripEventID'],
                             how='inner')


        #######

        union_all = pd.merge(union_all, self.BrakingIrregularity(df_copy),
                             on=['TripEventID'],
                             how='inner')

        # logging.info('checkpoint 7')
        # logging.info(f'{union_all.columns}')
        union_all.rename(columns={'DriverID_x': 'DriverID'}, inplace=True)
        # logging.info(f'{union_all.columns}')
        # logging.info(f'{union_all.shape}')
        union_all_groups = union_all.groupby(['CompanyID', 
                                                 'VehicleID', 
                                                 'DriverID', 
                                                 'TripEventID', 
                                                 'TripEventGuid', 
                                                 'CreatedDate']).agg({'AbsVal_TF_Moving_Delta':['mean','sum', 'std'],
                                                                                           'TripEventID':'last',
                                                                                           'ppeSpeed':'min', 
                                                                                           'AbsDiffBrakingForce':['mean', 'sum', 'std', 'max'],
                                                                                           'SpeedChange':'max', 
                                                                                           'DeltaSpeedBeforeZero':'max',
                                                                                           'DeltaSpeedAfterZero':'max', 
                                                                                           'DeltaSpeedRatio':'max',
                                                                                           'Max_Last_SecTimeSegment': 'max',
                                                                                           'AfterZeroBrakeOnCount':'first',
                                                                                           'Sum_CenterDifferenceDeviation':'max',
                                                                                           'Max_CenterDifferenceDeviation':['max','first'],
                                                                                           'Sum_TFCenterDeviation':'max',
                                                                                           'EndSpeedOverStartSpeed':'min',
                                                                                           'SIC':'max',
                                                                                           'Max_AbsValTurningForce':'first',
                                                                                           'Max_AbsValBrakingForce': 'first',
                                                                                           'Sum_TFCenterDeviation':'max',
                                                                                           'Max_TFCenterDeviation':'max',
                                                                                           'Sum_BFCenterDeviation':'max', 
                                                                                           'Max_BFCenterDeviation':'max',
                                                                                           'isSevere':'first'}
                                                                                           ).reset_index()


        union_all_groups.columns = [''.join(col).strip() for col in union_all_groups.columns.values]
        # logging.info('checkpoint 8 final')
        return union_all_groups
    
    def remove_duplicates(self, df):
        distinct_values = df[['TripEventID', 'VehicleID', 'EventTimeStamp']].drop_duplicates()
        distinct_values['EventTimeStamp'] = pd.to_datetime(distinct_values['EventTimeStamp'])
        distinct_values.sort_values(['VehicleID', 'EventTimeStamp'], inplace=True)
    
        filtered_indices = []  # To keep track of indices of rows to keep
    
        for vehicle_id, group in distinct_values.groupby('VehicleID'):
            last_time = pd.Timestamp.min  # Initialize with a very old date
    
            for index, row in group.iterrows():
                current_time = pd.Timestamp(row['EventTimeStamp'])
                if (current_time.to_pydatetime() - last_time.to_pydatetime()).total_seconds() > 60:
                    filtered_indices.append(index)
                    last_time = current_time
    
        # Filter the DataFrame to only include rows with indices in filtered_indices
        filtered_dataframe = distinct_values.loc[filtered_indices]
    
        df11 = df[df['TripEventID'].isin(filtered_dataframe['TripEventID'])]
    
        return df11
    
    def data_prediction(self, filename, data):
        try:
            if data is None or data.empty:
                logging.info(f'func_CMB_FCW_EB no data to predict')
            else:
                logging.info(f'{data.columns}')

                loaded_model = joblib.load(f'{filename}')

                columns_to_not_keep = ['DeltaSpeedAfterZeromax', 'DeltaSpeedBeforeZeromax', 'AbsVal_TF_Moving_Deltasum',
                                    'DeltaSpeedRatiomax', 'EndSpeedOverStartSpeedmin', 'ppeSpeedmin']
                data.drop(columns=columns_to_not_keep, inplace=True)

                data = data.dropna(subset=['AbsDiffBrakingForcestd'])


                columns_to_not_keep = ['CompanyID','DriverID', 'VehicleID', 'TripEventIDlast','TripEventID', 'TripEventGuid', 'CreatedDate','isSeverefirst']
                companyids = data['CompanyID']
                driverids = data['DriverID']
                vehicleid = data['VehicleID']
                tripeventids = data['TripEventID']
                tripeventguids = data['TripEventGuid']
                createddates = data['CreatedDate']
                

                data.drop(columns=columns_to_not_keep, inplace=True)
                # df_datatopredict_cleaned.drop(columns=['isCollisionfirst'], inplace=True)




                scaler = StandardScaler()

                data_scaled = scaler.fit_transform(data)

                ## look at the data here to find anything different to the other one

                predictions = loaded_model.predict(data_scaled)

                predictionsp = loaded_model.predict_proba(data_scaled)

                positive_probabilities = predictionsp[:, 1]

                # filtered_df_group[['CompanyID', 'VehicleID', 'TripEventID','TripEventGuid','CreatedDate', 'importancescore']]

                df_final = pd.DataFrame({'CompanyID': companyids, 'DriverID': driverids,'VehicleID': vehicleid, 'TripEventID':tripeventids, 
                                         'TripEventGuid': tripeventguids, 'CreatedDate': createddates, 'probabilities':positive_probabilities})
                # df_testpredict_feb14 = pd.DataFrame({'ids': ids, 'predictions': predictions, 'prob':positive_probabilities})
                # logging.info(f'{df_final.probabilities.describe()}')
                # df_final = df_final[df_final['probabilities'] >= 0.6]

                def parse_event_id(eventid):
                    # Parse the hexadecimal timestamp from the event ID.
                    return int(eventid[eventid.find('_') - 8: eventid.find('_')], 16)

                def calculate_importance(probability, eventid):
                    event_value = parse_event_id(eventid)

                    if probability < 0.6:
                        return 1 + ((probability*100 - 0) * (40 - 1) / (60 - 0))
                    elif probability >= 0.6 and event_value == 32:
                        return 50 + ((probability*100 - 60) * (95 - 50) / (100 - 60))
                    else:
                        return 70 + ((probability*100 - 60) * (95 - 70) / (100 - 60))
                    
                if df_final is None or df_final.empty:
                    return None, None
                else:
                
                    df_final['importancescore'] = df_final.apply(lambda row: calculate_importance(row['probabilities'], row['TripEventID']), axis=1)


                    df_importancescore = df_final[['CompanyID', 'VehicleID', 'TripEventID', 'TripEventGuid', 'CreatedDate', 'importancescore']].copy()

                    df_eventclassification = df_final[df_final.importancescore > 0]

                    # df_eventclassification['importancescore'] = df_eventclassification['importancescore']*100        

                    df_eventclassification['ClassificationName'] = df_eventclassification['importancescore'].apply(self.AER_helper.AER_classify_score) 

                    # logging.info(f'the importance scores are {df_eventclassification.importancescore} and the ClassificationName are {df_eventclassification.ClassificationName}')

                    
                    df_eventclassification.rename(columns={'CreatedDate': 'Timestamp'}, inplace=True)

                    df_eventclassification['By'] = 'AER Classification'

                    eventClassificaion_df = df_eventclassification[['CompanyID', 'VehicleID', 'DriverID', 'TripEventID','ClassificationName','Timestamp', 'By']]

                    df_importancescore['importancescore'] = df_importancescore['importancescore']/100
                    return df_importancescore, eventClassificaion_df


            
        except Exception as e:
            logging.error(f'func_CMB_FCW_EB Unable to predict TripEventID in {companyids[0]} because of {e}')
    
