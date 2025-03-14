import datetime
from datetime import datetime, timedelta

import pandas as pd
import sqlalchemy as sa
import numpy as np

from AER_helper import *
import AER_constants
import datetime
import logging

# from OverSpeed_helper import *
# import OverSpeed_constants

# SafetyDirect2_shard65
# Display the entire dataframe 
# pd.set_option('display.max_columns', None)

class Overspeed_Event_Classification:
    def __init__(self):
        self.ose_helper = AER_helperfunction()

    def set_initial_parameters(self, CIDList):
        companyid = CIDList[0]
        database = CIDList[1]
        shardid = CIDList[2]

        self.company_id = companyid
        self.databasename = database
        self.minutes_before_process = AER_constants.minutes_before_process
        self.current_date_time = datetime.datetime.utcnow()

        if(self.minutes_before_process is None):
            self.minutes_before_process = AER_constants.default_minutes_to_process

        # -- These variables establish the TripEvent CreatedDate range that will be processed.
	    # -- For example, if the "@minutesBeforeProcess" value is 30 (minutes), the dates will be the two most recent multiples of 30 minutes.
	    # -- Therefore, if this stored proc is executed at the datetime of "2023-01-01 08:02:00",
	    # --		then the "@beginCreatedDate" value will be "2023-01-01 07:30:00" and the "@endCreatedDate" value will be "2023-01-01 08:00:00".
	    # -- This establishes well-defined CreatedDate boundaries for processing data.
	    # -- In this situation, it is assumed that the job that executes this stored procedure will run every 30 minutes, or whatever the value of "@minutesBeforeProcess" is.

        minutes_before_event = (int((self.current_date_time - datetime.datetime.min).total_seconds() / 60 / self.minutes_before_process) * self.minutes_before_process)

        self.end_created_date = datetime.datetime.min + timedelta(minutes=minutes_before_event)
        self.begin_created_date = datetime.datetime.min + timedelta(minutes=minutes_before_event - self.minutes_before_process)

    def ose_get_data(self, server):
        engine = self.ose_helper.create_sql_connection(server)

        overspeed_event_dataframe = self.create_overspeed_event_dataframe(engine)
        # if not overspeed_event_dataframe.empty:
        #     logging.info(f'func_OSE over speed data frame returned with size {overspeed_event_dataframe.shape}')

        self.ose_helper.close_sql_connection()

        return overspeed_event_dataframe
    
    def nonsose_get_data(self, server):
        engine = self.ose_helper.create_sql_connection(server)

        nonsevere_overspeed_event_dataframe = self.create_nonsoverspeed_event_dataframe(engine)
        # if not overspeed_event_dataframe.empty:
        #     logging.info(f'func_OSE over speed data frame returned with size {overspeed_event_dataframe.shape}')

        self.ose_helper.close_sql_connection()

        return nonsevere_overspeed_event_dataframe
    
    def ose_update_data(self, upsert_trip_event_extra_dataframe, upsert_classification_dataframe, server):
        engine = self.ose_helper.create_sql_connection(server)

        self.ose_helper.add_importance_score_trip_event_extra(engine, upsert_trip_event_extra_dataframe, self.databasename)
        self.ose_helper.add_classification_type(engine, upsert_classification_dataframe, self.databasename)

        self.ose_helper.close_sql_connection()
    
    def ose_get_companydetails(self):
        engine = self.ose_helper.create_sql_connection()

        AutoEventClass_cids = self.ose_helper.AER_getcompany_database(engine)

        self.ose_helper.close_sql_connection()

        return AutoEventClass_cids
    
  


    def ose_prepare_data(self, overspeed_event_dataframe):
        columnname = 'ppespeedmph'
        if columnname  in overspeed_event_dataframe.columns:
            overspeed_event_dataframe['speedlimitmph'] = np.where(overspeed_event_dataframe['speedlimitmph'] % 5 <= 2, 
                                                              overspeed_event_dataframe['speedlimitmph'] - (overspeed_event_dataframe['speedlimitmph'] %  5), 
                                                              overspeed_event_dataframe['speedlimitmph'] + (5 - overspeed_event_dataframe['speedlimitmph'] % 5))


            overspeed_event_dataframe['invalid'] = np.where((overspeed_event_dataframe['speedlimitmph'] == 0) | 
                                                        (overspeed_event_dataframe['speedlimitmph'] > 85) |
                                                        (overspeed_event_dataframe['tespeedmph'] < overspeed_event_dataframe['speedlimitmph']), 1, 0)

        # overspeed_event_dataframe = self.remove_duplicate_over_speed_events_in_ose_dataframe(overspeed_event_dataframe)

        
            overspeed_event_dataframe['speedoverlimitmph'] = np.where(overspeed_event_dataframe['ppespeedmph'] < overspeed_event_dataframe['speedlimitmph'], 
                                                                0, 
                                                                overspeed_event_dataframe['ppespeedmph'] - overspeed_event_dataframe['speedlimitmph'])
            overspeed_event_dataframe['invalid'] = np.where((overspeed_event_dataframe['speedoverlimitmph'] == 0) |
                                                  (overspeed_event_dataframe['speedlimitmph'] == 0) |
                                                  ((overspeed_event_dataframe['speedoverlimitmph'] > 25) & (overspeed_event_dataframe['speedlimitmph'] < 40)), 1, 0)

            overspeed_event_dataframe = overspeed_event_dataframe.loc[overspeed_event_dataframe['invalid'] == 0]
        else:
            overspeed_event_dataframe['speedlimitmph'] = np.where(overspeed_event_dataframe['speedlimitmph'] % 5 <= 2, 
                                                              overspeed_event_dataframe['speedlimitmph'] - (overspeed_event_dataframe['speedlimitmph'] %  5), 
                                                              overspeed_event_dataframe['speedlimitmph'] + (5 - overspeed_event_dataframe['speedlimitmph'] % 5))
            # overspeed_event_dataframe['speedoverlimitmph'] = np.where(overspeed_event_dataframe['tespeedmph'] < overspeed_event_dataframe['speedlimitmph'], 
            #                                                     0, 
            #                                                     overspeed_event_dataframe['tespeedmph'] - overspeed_event_dataframe['speedlimitmph'])

            overspeed_event_dataframe['speedoverlimitmph'] = np.where(
                    overspeed_event_dataframe['speedlimitmph'] == 0,  # Check if speedlimit is 0
                        np.where(overspeed_event_dataframe['tespeedmph'] > 65, overspeed_event_dataframe['tespeedmph'] - 65,  # If speed > 65, subtract 65
                            np.where(overspeed_event_dataframe['tespeedmph'] > 55, overspeed_event_dataframe['tespeedmph'] - 55, 0)),  # If speed > 55, subtract 55, else 0
                    np.where(overspeed_event_dataframe['tespeedmph'] > overspeed_event_dataframe['speedlimitmph'], overspeed_event_dataframe['tespeedmph'] - overspeed_event_dataframe['speedlimitmph'], 0)  # If speed > speedlimit, subtract, else 0
                )
    
        

        return overspeed_event_dataframe[['CompanyID', 'VehicleID', 'DriverID', 'TripEventID', 'TripEventGuid', 'CreatedDate', 'speedoverlimitmph', 'invalid', 'importancescore']].copy()
    
    def nonsose_calculate_importance_score(self, overspeed_group_dataframe):
        overspeed_group_dataframe['importancescore'] = np.where(overspeed_group_dataframe['speedoverlimitmph'] > 0,
                                                                round((0.1 + (0.3/15) * (overspeed_group_dataframe['speedoverlimitmph'] - 0)), 2), 0)
        
        # overspeed_group_dataframe = overspeed_group_dataframe.groupby(['TripEventID'],group_keys=False).apply(lambda x: x[x['speedoverlimitmph'] == x['speedoverlimitmph'].max()]).drop_duplicates()

        ose = overspeed_group_dataframe[['CompanyID', 'VehicleID', 'TripEventID', 'TripEventGuid', 'CreatedDate', 'importancescore']].copy()

        overspeed_group_dataframe = overspeed_group_dataframe[overspeed_group_dataframe.importancescore > 0]

        overspeed_group_dataframe['importancescore'] = overspeed_group_dataframe['importancescore']*100        

        overspeed_group_dataframe['ClassificationName'] = overspeed_group_dataframe['importancescore'].apply(self.ose_helper.AER_classify_score) 

        # logging.info(f'the importance scores are {overspeed_group_dataframe.importancescore} and the ClassificationName are {overspeed_group_dataframe.ClassificationName}')

        
        overspeed_group_dataframe.rename(columns={'CreatedDate': 'Timestamp'}, inplace=True)

        overspeed_group_dataframe['By'] = 'AER Classification'

        eventClassificaion_df = overspeed_group_dataframe[['CompanyID', 'VehicleID', 'DriverID', 'TripEventID','ClassificationName','Timestamp', 'By']].copy()

        # print(ose)

        return ose, eventClassificaion_df

    def ose_calculate_importance_score(self, overspeed_group_dataframe):
        overspeed_group_dataframe['importancescore'] = np.where(overspeed_group_dataframe['speedoverlimitmph'] >= 5,
                                                                round((0.5 + (0.19/4.99) * (overspeed_group_dataframe['speedoverlimitmph'] - 5)), 2), 0)
        overspeed_group_dataframe['importancescore'] = np.where(overspeed_group_dataframe['speedoverlimitmph'] >= 10, 
                                                                round((0.7 + (0.24/4.99) * (overspeed_group_dataframe['speedoverlimitmph'] - 10)), 2), overspeed_group_dataframe['importancescore'])
        overspeed_group_dataframe['importancescore'] = np.where(overspeed_group_dataframe['speedoverlimitmph'] >= 15, 
                                                                round((0.95 + (0.05/19.00) * (overspeed_group_dataframe['speedoverlimitmph'] - 15)), 2), overspeed_group_dataframe['importancescore'])
        
        overspeed_group_dataframe = overspeed_group_dataframe.groupby(['TripEventID'],group_keys=False).apply(lambda x: x[x['speedoverlimitmph'] == x['speedoverlimitmph'].max()]).drop_duplicates()

        ose = overspeed_group_dataframe[['CompanyID', 'VehicleID', 'TripEventID', 'TripEventGuid', 'CreatedDate', 'importancescore']].copy()

        overspeed_group_dataframe = overspeed_group_dataframe[overspeed_group_dataframe.importancescore > 0]

        overspeed_group_dataframe['importancescore'] = overspeed_group_dataframe['importancescore']*100        

        overspeed_group_dataframe['ClassificationName'] = overspeed_group_dataframe['importancescore'].apply(self.ose_helper.AER_classify_score) 

        # logging.info(f'the importance scores are {overspeed_group_dataframe.importancescore} and the ClassificationName are {overspeed_group_dataframe.ClassificationName}')

        
        overspeed_group_dataframe.rename(columns={'CreatedDate': 'Timestamp'}, inplace=True)

        overspeed_group_dataframe['By'] = 'AER Classification'

        eventClassificaion_df = overspeed_group_dataframe[['CompanyID', 'VehicleID', 'DriverID', 'TripEventID','ClassificationName','Timestamp', 'By']].copy()

        # print(ose)

        return ose, eventClassificaion_df

 
        
    def create_overspeed_event_dataframe(self, engine):
        begin_created_date_string = self.ose_helper.get_datetime_sql_string(self.begin_created_date)
        end_created_date_string = self.ose_helper.get_datetime_sql_string(self.end_created_date)
        # begin_created_date_string = '2024-09-27 16:00:00'
        # end_created_date_string = '2024-09-27 20:00:00'
        logging.info("func_OSE inside creating dataframe")
        logging.info(f'time selected is from {begin_created_date_string} to {end_created_date_string}')

        ose_select_query = f"""
            SELECT
                te.CompanyID,
                te.VehicleID,
                te.TripID,
                te.CreatedDate,
                te.TripEventID,
                te.TripEventGuid as TripEventGuid,
                te.DriverID,
                e.EventName,
                te.Latitude,
                te.Longitude,
                te.SevereEvent,
                te.Speed * {AER_constants.convert_to_miles} as tespeedmph,
                round(te.SpeedLimit * {AER_constants.convert_to_miles}, 0) as speedlimitmph,
                0 AS speedoverlimitmph,
                te.Timestamp AS EventTimestamp,
                0 as isDup,
                0 as invalid,
                NULL as importancescore,
                round(p.Speed * {AER_constants.convert_to_miles}, 2) as ppespeedmph, 
                p.Timestamp AS PPETimestamp, 
                p.SequenceID
            FROM {self.databasename}.[dbo].[TripEvent] AS te
            JOIN [SafetyDirect2_master].[dbo].[Event] AS e WITH (NOLOCK)
                ON te.EventID = e.EventID
            JOIN {self.databasename}.[dbo].[PrePostEvent] as p with (NOLOCK)
                ON te.CompanyID = p.CompanyID
                AND te.VehicleID = p.VehicleID
            WHERE
                te.CompanyID = {self.company_id} 
                AND	te.EventID = {AER_constants.overspeed_event_id}
                AND	te.CreatedDate >= '{begin_created_date_string}'
                AND	te.CreatedDate < '{end_created_date_string}'
                AND p.Timestamp > DATEADD(second, 9, te.Timestamp)
				AND	p.Timestamp < DATEADD(second, 11, te.Timestamp)
            """
        
        # try:
        with engine.begin() as conn:
            overspeed_event_dataframe = pd.read_sql_query(sa.text(ose_select_query), conn)
            logging.info(f"func_OSE Data retrieved with size as {overspeed_event_dataframe.shape}")
            return overspeed_event_dataframe
        
    def create_nonsoverspeed_event_dataframe(self, engine):
        begin_created_date_string = self.ose_helper.get_datetime_sql_string(self.begin_created_date)
        end_created_date_string = self.ose_helper.get_datetime_sql_string(self.end_created_date)
        # begin_created_date_string = '2024-09-27 16:00:00'
        # end_created_date_string = '2024-09-27 20:00:00'
        logging.info("func_OSE inside creating dataframe")
        logging.info(f'time selected is from {begin_created_date_string} to {end_created_date_string}')

        ose_select_query = f"""
            SELECT
                te.CompanyID,
                te.VehicleID,
                te.TripID,
                te.CreatedDate,
                te.TripEventID,
                te.TripEventGuid as TripEventGuid,
                te.DriverID,
                te.Latitude,
                te.Longitude,
                te.SevereEvent,
                te.Speed * {AER_constants.convert_to_miles} as tespeedmph,
                round(te.SpeedLimit * {AER_constants.convert_to_miles}, 0) as speedlimitmph,
                0 AS speedoverlimitmph,
                te.Timestamp AS EventTimestamp,
                0 as isDup,
                0 as invalid,
                NULL as importancescore
            FROM {self.databasename}.[dbo].[TripEvent] AS te
            WHERE
                te.CompanyID = {self.company_id}
                AND te.SevereEvent = 0
                AND	te.EventID = {AER_constants.overspeed_event_id}
                AND	te.CreatedDate >= '{begin_created_date_string}'
                AND	te.CreatedDate < '{end_created_date_string}'
            """
        
        # try:
        with engine.begin() as conn:
            nonsoverspeed_event_dataframe = pd.read_sql_query(sa.text(ose_select_query), conn)
            logging.info(f"func_OSE Data retrieved with size as {nonsoverspeed_event_dataframe.shape}")
            return nonsoverspeed_event_dataframe
       
        
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