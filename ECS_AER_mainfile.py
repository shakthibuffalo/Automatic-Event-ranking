import datetime
from datetime import datetime, timedelta

import pandas as pd
import sqlalchemy as sa
import numpy as np
import logging
# from ECShelper import *
from AER_helper import *
import AER_constants
# import ECSconstants
import pywt
import datetime

from sklearn.preprocessing import StandardScaler
import joblib

class ECS_classification:

    def __init__(self):
        self.ECS_helper = ECS_Event_Helper()
        self.AER_helper = AER_helperfunction()

    def ECS_get_data(self, CIDList, server):
        engine = self.AER_helper.create_sql_connection(server)

        companyid = CIDList[0]
        database = CIDList[1]
        shardid = CIDList[2]

        ECS_event_dataframe = self.ECS_helper.create_ECS_event_dataframe(engine, database, companyid)

        self.AER_helper.close_sql_connection()

        return ECS_event_dataframe
    
    def DA_update_data(self, upsert_trip_event_extra_dataframe, upsert_classification_dataframe, CIDList, server):
        logging.info(f"func_ECS updating data with data size {upsert_trip_event_extra_dataframe.shape}")
        engine = self.AER_helper.create_sql_connection(server)

        companyid = CIDList[0]
        databasename = CIDList[1]
        shardid = CIDList[2]

        self.AER_helper.add_importance_score_trip_event_extra(engine, upsert_trip_event_extra_dataframe, databasename)
        self.AER_helper.add_classification_type(engine, upsert_classification_dataframe, databasename)
        self.AER_helper.close_sql_connection()
        logging.info("func_ECS Data updated")

    def DA_getautoeventclass_CID(self):
        engine = self.AER_helper.create_sql_connection()

        AutoEventClass_cids = self.AER_helper.AER_getcompany_database(engine)

        self.AER_helper.close_sql_connection()

        return AutoEventClass_cids
    

class ECS_Event_Helper:

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
    
    def create_ECS_event_dataframe(self, engine, database, companyid):

        self.set_initial_parameters()

        begin_created_date_string = self.begin_created_date.strftime("%Y-%m-%d %H:%M:%S")
        end_created_date_string = self.end_created_date.strftime("%Y-%m-%d %H:%M:%S")
       

        # begin_created_date_string = '2024-09-27 16:00:00'
        # end_created_date_string = '2024-09-27 20:00:00'
        logging.info("func_ECS inside creating dataframe")
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
                	te.speed as 'teSpeed',
                	te.TurningForce as 'teTurningForce',
                	te.curvature as 'teCurvature',
                	te.BrakingForce as 'teBrakingForce',
                	FORMAT(te.Timestamp, 'yyyy-MM-dd HH:mm:ss') as EventTimestamp,
                	tx.ImportanceScore as importancescore,
                	p.speed as ppeSpeed,
                  p.TurningForce as ppeTurningForce,
                  p.Curvature as ppeCurvature,
                  p.BrakingForce as ppeBrakingForce,
                  p.Timestamp as ppeTimeStamp
           	FROM {database}.dbo.TripEvent AS te
           	JOIN SafetyDirect2_master.dbo.Event AS e
               	ON te.EventID = e.EventID
           	LEFT JOIN {database}.dbo.TripEventExtra tx 
                  	ON te.CompanyID = tx.CompanyID
                  	AND te.VehicleID = tx.VehicleID
                  	AND te.TripEventID = tx.TripEventID
           	left join {database}.dbo.PrePostEvent p
                   	on te.CompanyID = p.CompanyID
                   	and te.VehicleID = p.VehicleID
           	WHERE te.CompanyID = {companyid}
                   AND	te.CreatedDate >= '{begin_created_date_string}'
                   AND	te.CreatedDate < '{end_created_date_string}'
                   	AND	 te.EventID = 1
                   	AND p.Timestamp > DATEADD(second, -10, te.Timestamp)
                       AND	p.Timestamp < DATEADD(second, 10, te.Timestamp)
                   	AND (tx.importancescore is null OR tx.importancescore = 0);
                """
        # try:
        with engine.begin() as conn:
            ECS_event_dataframe = pd.read_sql_query(sa.text(DA_select_query), conn)
            logging.info(f"func_ECS Data retrieved with size as {ECS_event_dataframe.shape}")
            return ECS_event_dataframe
        # except Exception as err:
        #     logging.error(f'func_ECS Error when trying to execute ECS_select_query: {err}')

        # DistanceAlert_event_dataframe = DistanceAlert_event_dataframe.assign(importancescore = 0)

    def remove_duplicates(self, df):
        distinct_values = df[['TripEventID', 'VehicleID', 'EventTimestamp']].drop_duplicates()
        distinct_values['EventTimestamp'] = pd.to_datetime(distinct_values['EventTimestamp'])
        distinct_values.sort_values(['VehicleID', 'EventTimestamp'], inplace=True)
    
        filtered_indices = []  # To keep track of indices of rows to keep
    
        for vehicle_id, group in distinct_values.groupby('VehicleID'):
            last_time = pd.Timestamp.min  # Initialize with a very old date
    
            for index, row in group.iterrows():
                current_time = pd.Timestamp(row['EventTimestamp'])
                if (current_time.to_pydatetime() - last_time.to_pydatetime()).total_seconds() > 60:
                    filtered_indices.append(index)
                    last_time = current_time
    
        # Filter the DataFrame to only include rows with indices in filtered_indices
        filtered_dataframe = distinct_values.loc[filtered_indices]
    
        df11 = df[df['TripEventID'].isin(filtered_dataframe['TripEventID'])]
    
        return df11
        
    
    def data_preperation(self, data):

        try:
        
            newdata = data[['TripEventID','CompanyID', 'VehicleID', 'TripEventGuid', 'CreatedDate', 'ppeSpeed', 'ppeTurningForce', 'ppeCurvature', 'ppeBrakingForce', 'ppeTimeStamp']]
            newdata = newdata.sort_values(by = ['TripEventID', 'ppeTimeStamp'], ascending = [True, True])
            
            grouped_data = newdata.groupby('TripEventID').agg({
                'ppeSpeed': ['mean', 'std', 'max', 'min'],
                'ppeTurningForce': ['mean', 'std', 'max', 'min'],
                'ppeCurvature': ['mean', 'std', 'max', 'min'],
                'ppeBrakingForce': ['mean', 'std', 'max', 'min']
            })

            # Flattening the multi-level column names
            grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]

            # Reset index to make 'masked_id' a column
            grouped_data.reset_index(inplace=True)
        except Exception as e:
            logging.error(f'func_ECS Unable to group data because {e}')

        # Function to apply wavelet transform and extract features from the ppeTurningForce signal
        def extract_wavelet_features(group):
            # Apply wavelet transform - using Daubechies wavelet
            coeffs = pywt.wavedec(group['ppeTurningForce'], 'db1', level=2)
            features = {}
            
            # Extract features from wavelet coefficients
            for i, coeff in enumerate(coeffs):
                features[f'wavelet_coeff_mean_{i}'] = np.mean(coeff)
                features[f'wavelet_coeff_std_{i}'] = np.std(coeff)
                features[f'wavelet_coeff_max_{i}'] = np.max(coeff)
                features[f'wavelet_coeff_min_{i}'] = np.min(coeff)

            return pd.Series(features)
        
        try:

            # Apply the wavelet transformation to each group of data per masked_id
            wavelet_features = newdata.groupby('TripEventID').apply(extract_wavelet_features)

            # Combine wavelet features with the original grouped data
            combined_data = pd.merge(grouped_data, wavelet_features, left_on='TripEventID', right_index=True)
            combined_data.rename(columns={'TripEventID_x': 'TripEventID'}, inplace=True)
            return combined_data
        except Exception as e:
            logging.error(f'func_ECS Unable to apply Wavelet transform because {e}')
    
    def data_prediction(self, filename, data):
        try:
            if data.empty:
                logging.info(f'func_ECS no data to predict')
            else:
                preddata = data.drop(['TripEventID'], axis=1)
            

                # Splitting the dataset into training and testing sets


                # Standardizing the features
                scaler = StandardScaler()
                preddata_scaled = scaler.fit_transform(preddata)

                loaded_model = joblib.load(f'{filename}')
                logging.info('func_ECS File Loaded successfully')
                
                predictions = loaded_model.predict(preddata_scaled)
                predictionsp = loaded_model.predict_proba(preddata_scaled)

                positive_probabilities = predictionsp[:, 1]
 
                data['predictions'] = predictions
                data['prob'] = positive_probabilities

                
                return data
        except Exception as e:
            logging.error(f'func_ECS Unable to predict TripEventID because of {e}')
    
    def ECS_claculate_importance_score(self, df):
        # filtered_df = df[df['prob'] >= 0.7]

        filtered_df_group = df.groupby(['CompanyID', 
                                                 'VehicleID', 
                                                 'DriverID', 
                                                 'TripEventID', 
                                                 'TripEventGuid', 
                                                 'CreatedDate']).agg(probability=('prob','first')).reset_index()

        # filtered_df_group = filtered_df.groupby(['CompanyID', 
        #                                          'VehicleID', 
        #                                          'DriverID', 
        #                                          'TripEventID', 
        #                                          'TripEventGuid', 
        #                                          'CreatedDate']).agg(teTurningForce=('teTurningForce','first'),
        #                                                              teSpeed=('teSpeed','first'),
        #                                                              avgSpeed=('ppeSpeed', 'mean')).reset_index()
        logging.info(f'func_ECS Predicted Amount of events that are True - {filtered_df_group.shape}')

        # def calculate_importance(probability):
        #     return 0.5 + ((probability - 0.8) * (0.95 - 0.5) / (1.0 - 0.8))
        
        def calculate_importance(probability):
            if probability >= 0.7:
                return 50 + ((probability*100 - 70) * (95 - 50) / (100 - 70))
            else:
                return 1 + ((probability*100 - 1) * (40 - 1) / (70 - 0))

        # Define a scoring function (this is a placeholder, adjust as needed)
        # def calculate_importance(turning_force, speed):
        #     return 70+(0.1*abs(turning_force) + 0.2*speed)
        # Apply the scoring function to each row
        # filtered_df_group['importancescore'] = filtered_df_group.apply(lambda row: calculate_importance(row['teTurningForce'], 
        #                                                                                     row['teSpeed']), axis=1)
        filtered_df_group['importancescore'] = filtered_df_group.apply(lambda row: calculate_importance(row['probability']), axis=1)
        
        filtered_df_group['ClassificationName'] = filtered_df_group['importancescore'].apply(self.AER_helper.AER_classify_score) 
        
        filtered_df_group['importancescore'] = filtered_df_group['importancescore']/100

        importancescore_df = filtered_df_group[['CompanyID', 'VehicleID', 'TripEventID','TripEventGuid','CreatedDate', 'importancescore']]

        filtered_df_group.rename(columns={'CreatedDate': 'Timestamp'}, inplace=True)

        filtered_df_group['By'] = 'AER Classification'

        eventClassificaion_df = filtered_df_group[['CompanyID', 'VehicleID', 'DriverID', 'TripEventID','ClassificationName','Timestamp', 'By']]


        return importancescore_df, eventClassificaion_df
    


    