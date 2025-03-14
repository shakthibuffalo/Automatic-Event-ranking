import pandas as pd
import datetime
import sqlalchemy as sa
import numpy as np
import datetime

import logging
import os
import AER_helper
import AER_constants

from datetime import datetime, timedelta


####################################################################################################################
class AER_WeeklyReportMain:

    def __init__(self):
        self.Report_helper = AER_weeklyreport()
        self.AER_helper = AER_helper.AER_helperfunction()


    def AER_get_data(self, database, companyid, server):
        engine = self.AER_helper.create_sql_connection(server)

        ECS_event_dataframe, begindate, enddate, cid = self.Report_helper.create_WeeklyStats_dataframe(engine, database, companyid)

        self.AER_helper.close_sql_connection()

        logging.info(f"retrieved with size as {ECS_event_dataframe.shape}, {begindate}, {enddate}, {cid}")

        return ECS_event_dataframe, begindate, enddate, cid
    
    def AER_getautoeventclass_CID(self):
        engine = self.AER_helper.create_sql_connection()

        AutoEventClass_cids = self.AER_helper.AER_getcompany_database(engine)

        self.AER_helper.close_sql_connection()

        return AutoEventClass_cids
    

class AER_weeklyreport:

    def set_initial_parameters(self):

        self.today = datetime.now()

        # Determine how many days to subtract to get the most recent Sunday
        days_to_subtract = (self.today.weekday() + 1) % 7

        # Calculate the most recent Sunday date
        self.enddate = self.today - timedelta(days=days_to_subtract)

        # Calculate the Sunday before the most recent Sunday
        self.begindate = self.enddate - timedelta(days=7)
        
        # self.minutes_before_process = AER_constants.minutes_before_process
        # self.current_date_time = datetime.datetime.utcnow()
        # self.current_date_time = self.current_date_time

        # if(self.minutes_before_process is None):
        #     self.minutes_before_process = AER_constants.default_minutes_to_process

        # # -- These variables establish the TripEvent CreatedDate range that will be processed.
	    # # -- For example, if the "@minutesBeforeProcess" value is 30 (minutes), the dates will be the two most recent multiples of 30 minutes.
	    # # -- Therefore, if this stored proc is executed at the datetime of "2023-01-01 08:02:00",
	    # # --		then the "@beginCreatedDate" value will be "2023-01-01 07:30:00" and the "@endCreatedDate" value will be "2023-01-01 08:00:00".
	    # # -- This establishes well-defined CreatedDate boundaries for processing data.
	    # # -- In this situation, it is assumed that the job that executes this stored procedure will run every 30 minutes, or whatever the value of "@minutesBeforeProcess" is.

        # minutes_before_event = (int((self.current_date_time - datetime.datetime.min).total_seconds() / 60 / self.minutes_before_process) * self.minutes_before_process)

        # self.end_created_date = datetime.datetime.min + datetime.timedelta(minutes=minutes_before_event)
        # self.begin_created_date = datetime.datetime.min + datetime.timedelta(minutes=minutes_before_event - self.minutes_before_process)

    def create_WeeklyStats_dataframe(self, engine, database, companyid):
        self.set_initial_parameters()

        # Format dates as 'yyyy-MM-dd'
        end_created_date_string = self.enddate.strftime('%Y-%m-%d')
        begin_created_date_string = self.begindate.strftime('%Y-%m-%d')
        

        # begin_creatiskon_date_string = '2024-05-22 23:30:00'
        logging.info(f'time selected is from {begin_created_date_string} to {end_created_date_string}')
        logging.info("func_CMB_FCW_EB inside creating dataframe")

       
        DA_select_query = f"""
            Select a.Name as 'Group', a.EventName, Count(*) as 'Count' from
                (Select te.TripEventID, te.ImportanceScore, t.EventID, ev.EventName, ec.Classification, e.Name from
                {database}.dbo.TripEventExtra te 
                    left join {database}.dbo.TripEvent t on te.TripEventID = t.TripEventID
                    left join SafetyDirect2_master.dbo.Event ev on t.EventID = ev.EventID
                    left join {database}.dbo.EventClassification ec on te.TripEventID = ec.TripEventID
                    left join {database}.dbo.EventClassificationType e on ec.CompanyID = e.CompanyID and ec.Classification = e.Type
                where e.Name in ('Pre-Grouped (Low)', 'Pre-Grouped (Mid)', 'Pre-Grouped (High)') and te.CompanyID = {companyid} and te.CreatedDate >= '{begin_created_date_string}' and te.CreatedDate < '{end_created_date_string}'
                ) a 
                where a.EventID is not null group by a.Name, a.EventName
        """
        logging.info(f'{DA_select_query}')
        # try:
        with engine.begin() as conn:
            CMBFCWEB_event_dataframe = pd.read_sql_query(sa.text(DA_select_query), conn)
        # except Exception as err:
        #     raise Exception(f"Error when trying to execute DA_select_query: {err}")

        if CMBFCWEB_event_dataframe is None or CMBFCWEB_event_dataframe.empty:
            logging.info(f'func_CMB_FCW_EB no data from {begin_created_date_string} to {end_created_date_string}')
            return None
        else:

            logging.info(f"func_CMB_FCW_EB Data retrieved with size as {CMBFCWEB_event_dataframe.shape}, {begin_created_date_string}, {end_created_date_string}, {companyid}")
            return CMBFCWEB_event_dataframe, begin_created_date_string, end_created_date_string, companyid