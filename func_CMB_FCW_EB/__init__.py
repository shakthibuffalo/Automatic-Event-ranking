# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging

import AER_constants
from AER_helper import *
import CMB_FCW_EB_helper
import pandas as pd

def main(name: str) -> str:

    CMB_classifier = CMB_FCW_EB_helper.CMB_FCW_EB_Classification()
    CMB_dataret = CMB_FCW_EB_helper.CMB_FCW_EB()
    CMB_helper = CMB_FCW_EB_helper.data_preperation()
    AER_helper = AER_helperfunction()

    filename = r'best_xgboost_model.pkl'

    servername = AER_helper.AER_getserveripmapping(name, AER_constants.DataSource_mapping_dict)
    logging.info(f'the server name is {servername}')

    if servername == 'DbServer':
        
        try:
            CMB_Dataframe = pd.DataFrame()
            CMB_Dataframe = CMB_classifier.CMBFCWEB_get_data(name, 'DbServer')


            if CMB_Dataframe is not None and not CMB_Dataframe.empty:
                import sklearn
                logging.info(f'{sklearn.__version__}')
                import numpy
                logging.info(f'numpy version: {numpy.__version__}')
                df = CMB_Dataframe.drop_duplicates(subset=['TripEventID']).copy()
                
                CMB_Dataframeshape = df.shape if CMB_Dataframe is not None else (0,0)
                CMB_Dataframe_nodup = CMB_helper.remove_duplicates(CMB_Dataframe)
                df1 = CMB_Dataframe_nodup.drop_duplicates(subset=['TripEventID']).copy()

                CMB_Dataframe_nodupshape = df1.shape

                CMB_Dataframe_cleaned = CMB_helper.feature_generation(CMB_Dataframe)

                CMB_Dataframe_cleanedshape = CMB_Dataframe_cleaned.shape

                CMB_Datafreane_predicted, CMB_Datadrame_classofied = CMB_helper.data_prediction(filename, CMB_Dataframe_cleaned)
                CMB_Datafreane_predictedshape = CMB_Datafreane_predicted.shape
                jsonResult = CMB_Datafreane_predicted.head().to_json(orient='records')
                final_output = {
                    "header": name,
                    "data": json.loads(jsonResult)
                }
                

                if CMB_Datafreane_predicted is not None and not CMB_Datafreane_predicted.empty:
                    CMB_classifier.CMBFCWEB_update_data(CMB_Datafreane_predicted, CMB_Datadrame_classofied, name, 'DbServer')

                return json.dumps(final_output)
                
            else:
                CMB_Dataframeshape = (0,0)
                CMB_Dataframe_nodupshape=(0,0)
                CMB_Dataframe_cleanedshape =(0,0)
                CMB_Datafreane_predictedshape = (0,0)

            logging.info(f'the data stats are = {CMB_Dataframeshape[0]}, {CMB_Dataframe_nodupshape[0]}, {CMB_Dataframe_cleanedshape[0]}, {CMB_Datafreane_predictedshape[0]}')
        except Exception as e:
            logging.error(f'func_CMB exception for {name} as {e}')
            

    elif servername == 'Db2Server':
        
        try:
            CMB_Dataframe = pd.DataFrame()
            CMB_Dataframe = CMB_classifier.CMBFCWEB_get_data(name, 'Db2Server')


            if CMB_Dataframe is not None and not CMB_Dataframe.empty:
                import sklearn
                logging.info(f'{sklearn.__version__}')
                import numpy
                logging.info(f'numpy version: {numpy.__version__}')
                df = CMB_Dataframe.drop_duplicates(subset=['TripEventID']).copy()
                
                CMB_Dataframeshape = df.shape
                CMB_Dataframe_nodup = CMB_helper.remove_duplicates(CMB_Dataframe)
                df1 = CMB_Dataframe_nodup.drop_duplicates(subset=['TripEventID']).copy()
                
                CMB_Dataframe_nodupshape = df1.shape

                CMB_Dataframe_cleaned = CMB_helper.feature_generation(CMB_Dataframe)
                CMB_Dataframe_cleanedshape = CMB_Dataframe_cleaned.shape

                CMB_Datafreane_predicted, CMB_Datadrame_classofied = CMB_helper.data_prediction(filename, CMB_Dataframe_cleaned)
                CMB_Datafreane_predictedshape = CMB_Datafreane_predicted.shape

                jsonResult = CMB_Datafreane_predicted.head().to_json(orient='records')
                final_output = {
                    "header": name,
                    "data": json.loads(jsonResult)
                }
                

                if CMB_Datafreane_predicted is not None and not CMB_Datafreane_predicted.empty:
                    CMB_classifier.CMBFCWEB_update_data(CMB_Datafreane_predicted, CMB_Datadrame_classofied, name, 'Db2Server')

                return json.dumps(final_output)
            else:
                CMB_Dataframeshape =(0,0)
                CMB_Dataframe_nodupshape=(0,0)
                CMB_Dataframe_cleanedshape =(0,0)
                CMB_Datafreane_predictedshape = (0,0)
            logging.info(f'the data stats are = {CMB_Dataframeshape[0]}, {CMB_Dataframe_nodupshape[0]}, {CMB_Dataframe_cleanedshape[0]}, {CMB_Datafreane_predictedshape[0]}')

        except Exception as e:
            logging.error(f'func_CMB exception for {name} as {e}')
            
    else:
        logging.info('no server')
    
