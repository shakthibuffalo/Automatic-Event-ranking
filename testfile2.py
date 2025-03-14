import datetime
from datetime import datetime, timedelta

import pandas as pd
import sqlalchemy as sa
import numpy as np
import logging
# from DAhelper import *
import datetime
# import DAconstsants
import AER_constants
from AER_helper import *
class Distance_alert_classification:

    def __init__(self):
        self.DA_helper = Distance_alert_helper()
        self.AER_helper = AER_helperfunction()

    def DA_get_data(self, CIDList, server):
        engine = self.AER_helper.create_sql_connection(server)

        companyid = CIDList[0]
        database = CIDList[1]
        shardid = CIDList[2]

        DistanceAlert_event_dataframe = self.DA_helper.create_DistanceAlert_event_dataframe(engine, f'{database}', companyid)

        self.AER_helper.close_sql_connection()

        return DistanceAlert_event_dataframe
    
    def DA_update_data(self, upsert_trip_event_extra_dataframe, upsert_classification_dataframe,  CIDList, server):
        # logging.info(f"updating data with data size {upsert_trip_event_extra_dataframe.shape}")
        engine = self.AER_helper.create_sql_connection(server)

        companyid = CIDList[0]
        databasename = CIDList[1]
        shardid = CIDList[2]

        self.AER_helper.add_importance_score_trip_event_extra(engine, upsert_trip_event_extra_dataframe, databasename)
        self.AER_helper.add_classification_type(engine, upsert_classification_dataframe, databasename)

        self.AER_helper.close_sql_connection()
        logging.info(f"func_DA for companyid {companyid} Data updated")

    def DA_getautoeventclass_CID(self):
        engine = self.AER_helper.create_sql_connection()

        AutoEventClass_cids = self.AER_helper.AER_getcompany_database(engine)

        self.AER_helper.close_sql_connection()

        return AutoEventClass_cids
    
class Distance_alert_helper:

    def __init__(self):
        self.AER_helper = AER_helperfunction()

    def set_initial_parameters(self):
        
        self.minutes_before_process = AER_constants.minutes_before_process
        self.current_date_time = datetime.datetime.utcnow()
        self.current_date_time = self.current_date_time

        if(self.minutes_before_process is None):
            self.minutes_before_process = AER_constants.default_minutes_to_process

        # -- These variables establish the TripEvent CreatedDate range that will be processed.
	    # -- For example, if the "@minutesBeforeProcess" value is 30 (minutes), the dates will be the two most recent multiples of 30 minutes.
	    # -- Therefore, if this stored proc is executed at the datetime of "2023-01-01 08:02:00",
	    # --		then the "@beginCreatedDate" value will be "2023-01-01 07:30:00" and the "@endCreatedDate" value will be "2023-01-01 08:00:00".
	    # -- This establishes well-defined CreatedDate boundaries for processing data.
	    # -- In this situation, it is assumed that the job that executes this stored procedure will run every 30 minutes, or whatever the value of "@minutesBeforeProcess" is.

        minutes_before_event = (int((self.current_date_time - datetime.datetime.min).total_seconds() / 60 / self.minutes_before_process) * self.minutes_before_process)

        self.end_created_date = datetime.datetime.min + datetime.timedelta(minutes=minutes_before_event)
        self.begin_created_date = datetime.datetime.min + datetime.timedelta(minutes=minutes_before_event - self.minutes_before_process)

    def create_DistanceAlert_event_dataframe(self, engine, database, companyid):
        logging.info("inside creating dataframe")

        self.set_initial_parameters()

        begin_created_date_string = self.begin_created_date.strftime("%Y-%m-%d %H:%M:%S")
        end_created_date_string = self.end_created_date.strftime("%Y-%m-%d %H:%M:%S")
       

        # begin_created_date_string = '2024-09-27 16:00:00'
        # end_created_date_string = '2024-09-27 20:00:00'
        logging.info("func_DA inside creating dataframe")
        logging.info(f'time selected is from {begin_created_date_string} to {end_created_date_string}')

        DA_select_query = f"""
            select
                te.CompanyID,
                te.VehicleID,
                FORMAT(te.CreatedDate, 'yyyy-MM-dd') as CreatedDate,
                te.TripEventID,
                te.TripEventGuid,
                te.DriverID,
                e.EventName,
                te.SevereEvent,
                te.speed as teSpeed,
                pe.speed as ppeSpeed,
                te.SpeedForwardVehicle as teSpeedforwardV,
                pe.SpeedForwardVehicle as ppeSpeedforwardV,
                te.DistanceForwardVehicle as teDistanceforwardV,
                pe.DistanceForwardVehicle as ppeDistanceforwardV,
                FORMAT(te.Timestamp, 'yyyy-MM-dd HH:mm:ss') as EventTimestamp,
                FORMAT(pe.Timestamp, 'yyyy-MM-dd HH:mm:ss') as ppeTimeStamp,
                CONVERT(varchar, DATEDIFF(second, te.Timestamp, pe.Timestamp)) + '.' + CONVERT(varchar, pe.SequenceID) as 'TimeSegment',
                DATEDIFF(second, te.Timestamp, pe.Timestamp) + ((pe.SequenceID - 1) * 0.25) as 'SecTimeSegment',     
                tx.ImportanceScore as importancescore
                FROM {database}.dbo.TripEvent AS te
                JOIN SafetyDirect2_master.dbo.Event AS e
                    ON te.EventID = e.EventID
                LEFT JOIN {database}.dbo.TripEventExtra tx 
                    ON te.CompanyID = tx.CompanyID
                    AND te.VehicleID = tx.VehicleID
                    AND te.TripEventID = tx.TripEventID
                left join {database}.dbo.PrePostEvent pe
                    ON te.CompanyID = pe.CompanyID
                    AND te.VehicleID = pe.VehicleID
                WHERE	te.CompanyID = {companyid}
                    AND	 te.CreatedDate >= '{begin_created_date_string}'
                    AND	te.CreatedDate < '{end_created_date_string}'
                    AND	 te.EventID = {AER_constants.DA_event_id}
                    AND pe.Timestamp > DATEADD(second, -10, te.Timestamp)
                    AND	pe.Timestamp < DATEADD(second, 10, te.Timestamp)
                    AND tx.importancescore is null;

        """
        with engine.begin() as conn:
            DistanceAlert_event_dataframe = pd.read_sql_query(sa.text(DA_select_query), conn)
        # except Exception as err:
        #     raise Exception(f"Error when trying to execute DA_select_query: {err}")

        # DistanceAlert_event_dataframe = DistanceAlert_event_dataframe.assign(importancescore = 0)
        if DistanceAlert_event_dataframe is None or DistanceAlert_event_dataframe.empty:
            logging.info(f'Func_da no data from {begin_created_date_string} to {end_created_date_string}')
            return None
        else:
            logging.info(f"Data retrieved with size as {DistanceAlert_event_dataframe.shape}")
            return DistanceAlert_event_dataframe
    
    def remove_noise(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        required_columns = ['TripEventID', 'ppeDistanceforwardV', 
                            'ppeSpeedforwardV', 'ppeSpeed']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        grouped = data.groupby('TripEventID')
        clean_data_list = []

        for name, group in grouped:
            # Mark rows where the speed and distance of the forward vehicle are zero
            group['zero_forward_speed'] = (group['ppeSpeedforwardV'] == 0) | (group['ppeSpeedforwardV'] > 230)
            group['zero_forward_distance'] = (group['ppeDistanceforwardV'] == 0)
            group['current_speed_not_zero'] = (group['ppeSpeed'] > 0)

            # Check conditions simultaneously
            group['zero_both'] = group['zero_forward_speed'] & group['zero_forward_distance'] & group['current_speed_not_zero']
            
            # Find stretches of at least four consecutive zero readings (1 second)
            group['zero_block'] = (group['zero_both'].rolling(window=2, min_periods=2).sum() == 2)
            
            # Extend this block to cover the entire second of zero readings
            group['to_remove'] = group['zero_block'].replace({False: 0}).replace(to_replace=0, method='ffill', limit=2)
            
            # Append non-noisy data to the clean dataset
            # clean_data = clean_data.append(group[group['to_remove'] == 0].drop(columns=['zero_forward_speed', 'zero_forward_distance', 'zero_both', 'zero_block', 'to_remove']), ignore_index=True)
            clean_data_list.append(group[group['to_remove'] == 0].drop(columns=['zero_forward_speed', 'zero_forward_distance', 'current_speed_not_zero', 'zero_both', 'zero_block', 'to_remove']))
            
        clean_data = pd.concat(clean_data_list, ignore_index=True)
        return clean_data


    # Function to identify outliers based on the IQR
    def get_outlier_mask(self, group):
        masks = {}
        for column in ['ppeSpeedforwardV', 'ppeDistanceforwardV']:
            Q1 = group[column].quantile(0.25)
            Q3 = group[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Create a mask for each column where True indicates an outlier
            masks[column] = (group[column] < lower_bound) | (group[column] > upper_bound)
        
        # Combine masks to identify rows where both conditions are true
        combined_outlier_mask = masks['ppeSpeedforwardV'] & masks['ppeDistanceforwardV']
        return ~combined_outlier_mask
    
    def normalize(self, series):
    
        return (series - series.min()) / (series.max() - series.min())
    

    def process_events(self, df):
        df['EventTimestamp'] = pd.to_datetime(df['EventTimestamp'])
        df_sorted = df.sort_values(by=['CompanyID', 'DriverID', 'EventTimestamp', 'ppeTimeStamp'])

        """
        Calculate scores for events based on the distance to the forward vehicle, 
        the speed of the current vehicle, and the speed of the forward vehicle.

        Parameters:
        df (pd.DataFrame): DataFrame containing the event data with columns 'eventid', 
                        'distance_to_forward_vehicle', 'speed_of_current_vehicle', 
                        'speed_of_forward_vehicle'.

        Returns:
        pd.DataFrame: DataFrame with event scores.
        """
        # Normalize the distance, speed of current vehicle, and speed of forward vehicle for each event
        df['normalized_distance'] = df.groupby('TripEventID')['ppeDistanceforwardV'].transform(self.normalize)
        df['normalized_speed_current'] = df.groupby('TripEventID')['ppeSpeed'].transform(self.normalize)
        df['normalized_speed_forward'] = df.groupby('TripEventID')['ppeSpeedforwardV'].transform(self.normalize)

        # Invert the normalized distance to get higher scores for closer distances
        df['inverted_normalized_distance'] = 1 - df['normalized_distance']

        # Calculate mean and standard deviation for each event
        aggregation = df.groupby('TripEventID').agg({
            'inverted_normalized_distance': ['mean', 'std'],
            'normalized_speed_current': ['mean', 'std'],
            'normalized_speed_forward': ['mean', 'std']
        })

        # Flatten MultiIndex columns
        aggregation.columns = ['_'.join(col) for col in aggregation.columns]

        # Calculate a composite score for each event
        # Higher score for being consistently close
        aggregation['score'] = ((aggregation['inverted_normalized_distance_mean']) * 0.7 +
                                aggregation['normalized_speed_current_mean'] * 0.2 +
                                aggregation['normalized_speed_forward_mean'] * 0.1)

        return aggregation.reset_index()
    
    def calculate_scores(self, df):
        if df is None or df.empty:
            logging.info(f'func_DA no data to predict')
            return None, None
        
        else:
            try:

                df['EventTimestamp'] = pd.to_datetime(df['EventTimestamp'])
                df_sorted = df.sort_values(by=['CompanyID', 'DriverID', 'EventTimestamp', 'ppeTimeStamp'])

                df_clean = self.remove_noise(df_sorted)
                df_cleaned = df_clean.groupby('TripEventID').apply(lambda group: group[self.get_outlier_mask(group)]).reset_index(drop=True)

                event_scores = self.process_events(df_cleaned)

                df1 = event_scores[['TripEventID', 'score']]

                df1 = df1.dropna(subset=['score'])

                columns_to_select = ['TripEventID','CompanyID', 'VehicleID', 'DriverID', 'TripEventGuid', 'CreatedDate']
                # df2 = df_cleaned[['CompanyID', 'VehicleID', 'DriverID', 'TripEventID', 'TripEventGuid', 'CreatedDate']]

                df2 = df_cleaned.groupby('TripEventID')[columns_to_select[1:]].first().reset_index()

                merged_df = pd.merge(df1, df2, on='TripEventID', how='left')

                merged_df['classificationscore'] = merged_df['score']*100

                merged_df['ClassificationName'] = merged_df['classificationscore'].apply(self.AER_helper.AER_classify_score)

                merged_df.rename(columns={'score': 'importancescore'}, inplace=True)

                importancescore_df = merged_df[['CompanyID', 'VehicleID', 'TripEventID','TripEventGuid','CreatedDate', 'importancescore']]

                merged_df.rename(columns={'CreatedDate': 'Timestamp'}, inplace=True)

                merged_df['By'] = 'AER Classification'

                eventClassificaion_df = merged_df[['CompanyID', 'VehicleID', 'DriverID', 'TripEventID','ClassificationName','Timestamp', 'By']]

                return importancescore_df, eventClassificaion_df
            
            except Exception as e:
                logging.error(f'func_DA Unable to predict TripEventID because of {e}') 






    
    




    # def DA_prepare_data(self, df):
    #     logging.info(f"inside data preperation with data size as {df.shape}")
    #     df['EventTimestamp'] = pd.to_datetime(df['EventTimestamp'])
    #     df_sorted = df.sort_values(by=['CompanyID', 'DriverID', 'EventTimestamp'])

    #     df_result = df_sorted[df_sorted['DriverID']!=""]

    #     df_result = df_result[df_result.notna()]
        
    #     # nullvalues = df_result.isna().sum()

    #     # if nullvalues.all() == 0:
    #     df_tripevent = df_result.filter(['CompanyID', 'DriverID', 'EventTimestamp', 'TripEventID','VehicleID','TripEventGuid', 'importancescore', 'CreatedDate'], axis=1)

        
    #     df_tripevent['EventTimestamp'] = df_tripevent['EventTimestamp'].dt.date
        
    #     df_tripevent['importancescore'].replace('None', '0.0', inplace=True)
    #     df_tripevent['importancescore'] = df_tripevent['importancescore'].astype(float)
    #     df_tripevent['importancescore'] = df_tripevent['importancescore'].fillna(value=0)
        
        
    #     df_tripevent = df_tripevent.sort_values(by=['CompanyID', 'DriverID', 'EventTimestamp'])
    #     df_tripevent['EventTimestamp'] = pd.to_datetime(df_tripevent['EventTimestamp'])

    #     df_tripevent['newimportancescore'] = np.where((df_tripevent['DriverID'] == df_tripevent['DriverID'].shift(1)) & 
    #                                                    (df_tripevent['importancescore'].shift(1) != 0) &
    #                                                    (df_tripevent['importancescore'] == 0) &
    #                                                    ((df_tripevent['EventTimestamp'] - df_tripevent['EventTimestamp'].shift(1)).dt.days < 7), 
    #                                                    df_tripevent['importancescore'].shift(1)+0.05, 0.0)

    #     df_tripevent['newimportancescore'] = df_tripevent['newimportancescore'].clip(upper=1)
        
        
    #     df_tripevent_oldupdates = df_tripevent[(df_tripevent['newimportancescore'] != 0.0) & (df_tripevent['newimportancescore'].notna() == True)]
        
    #     df_tripevent = df_tripevent[(df_tripevent['importancescore'] == 0.0) & (df_tripevent['newimportancescore'] == 0.0)]
    #     df_tripevent = df_tripevent.drop('newimportancescore', axis=1)

        
        
        
    #     # old_update_data = list(zip(df_tripevent_oldupdates.CompanyID, df_tripevent_oldupdates.VehicleID, df_tripevent_oldupdates.TripEventID, df_tripevent_oldupdates.TripEventGuid, df_tripevent_oldupdates.CreatedDate, df_tripevent_oldupdates.newimportancescore))

    #     old_update_data = df_tripevent_oldupdates[['CompanyID', 'VehicleID', 'DriverID','TripEventID', 'TripEventGuid', 'CreatedDate', 'newimportancescore']]
    #     old_update_data = old_update_data.rename({'newimportancescore': 'importancescore'}, axis=1)

    #     old_importancescore_df = old_update_data[['CompanyID', 'VehicleID','TripEventID', 'TripEventGuid', 'CreatedDate', 'importancescore']]

    #     old_update_data['importancescore'] = old_update_data['importancescore']*100

    #     old_update_data['Classification'] = old_update_data['importancescore'].apply(self.AER_helper.AER_classify_score) 

    #     old_update_data.rename(columns={'CreatedDate': 'Timestamp'}, inplace=True)

    #     old_update_data['By'] = 'AER Classification'

    #     old_eventClassificaion_df = old_update_data[['CompanyID', 'VehicleID', 'DriverID', 'TripEventID','Classification','Timestamp', 'By']]

        

    #     v = df_tripevent.DriverID.value_counts()
        
        

    #     df_tripevent_1 = df_tripevent[df_tripevent.DriverID.isin(v.index[v.gt(2)])]

        
    #     logging.info(f"last line of data preperation function with final data size as {df_tripevent_1.shape}")
        
    #     return df_tripevent_1, old_importancescore_df, old_eventClassificaion_df

    # def calculate_importance_score(self, df):
    #     # Function to normalize scores
    #     df = df.reset_index()
    #     def normalize_scores(scores, min_val, max_val):
    #         if min_val == max_val:
    #             return [1.0 if score > 0 else 0 for score in scores]
    #         return [(0.7 + (score - min_val) * 0.3 / (max_val - min_val)) if score > 0 else 0 for score in scores]

    #     # Convert EventTimestamp to datetime
    #     df['EventTimestamp'] = pd.to_datetime(df['EventTimestamp'])

    #     # Sort by DriverID and EventTimestamp
    #     df.sort_values(by=['DriverID', 'EventTimestamp'], inplace=True)

    #     # Dictionary to store scores, using DataFrame index as keys
    #     scores = {index: 0 for index in df.index}

    #     # Iterate over each driver
    #     for driver in df['DriverID'].unique():
    #         driver_events = df[df['DriverID'] == driver]
    #         event_dates = driver_events['EventTimestamp'].tolist()
    #         count = 0

    #         # Iterate over the events
    #         for i in range(len(event_dates)):
    #             if i >= 2 and event_dates[i] - event_dates[i - 2] <= timedelta(days=7):
    #                 count += 1
    #                 scores[driver_events.index[i]] = count
    #             elif i > 0 and event_dates[i] - event_dates[i - 1] > timedelta(days=7):
    #                 count = 0

    #     # Extract scores from the dictionary and normalize them
    #     score_list = list(scores.values())
    #     if not score_list or all(score == 0 for score in score_list):
    #         df['importancescore'] = 0
    #     else:
    #         max_score = max(score_list)
    #         min_score = min(filter(lambda x: x > 0, score_list))
    #         normalized_scores = normalize_scores(score_list, min_score, max_score)

    #         for index in df.index:
    #             df.at[index, 'importancescore'] = normalized_scores[index - df.index[0]]

    #     df = df[df.importancescore > 0]

    #     importancescore_df = df[['CompanyID', 'VehicleID', 'TripEventID','TripEventGuid','CreatedDate', 'importancescore']]

    #     df['importancescore'] = df['importancescore']*100        

    #     df['ClassificationName'] = df['importancescore'].apply(self.AER_helper.AER_classify_score) 

    #     logging.info(f'the importance scores are {df.importancescore} and the ClassificationName are {df.ClassificationName}')

        
    #     df.rename(columns={'CreatedDate': 'Timestamp'}, inplace=True)

    #     df['By'] = 'AER Classification'

    #     eventClassificaion_df = df[['CompanyID', 'VehicleID', 'DriverID', 'TripEventID','ClassificationName','Timestamp', 'By']]

    #     return importancescore_df, eventClassificaion_df

    