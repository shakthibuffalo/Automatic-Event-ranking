# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging

import ECS_AER_mainfile
from AER_helper import *
import AER_constants
import pandas as pd

def main(name: str) -> str:

    logging.info(f'Processing name {name}')

    ECS = ECS_AER_mainfile.ECS_classification()
    ECS_helper = ECS_AER_mainfile.ECS_Event_Helper()
    AER_helper = AER_helperfunction()
    filename = r'optimized_rf_model3.pkl'

    servername = AER_helper.AER_getserveripmapping(name, AER_constants.DataSource_mapping_dict)

    if servername == 'DbServer':
        try:
            ECS_dataframe = ECS.ECS_get_data(name, 'DbServer')
            if ECS_dataframe is not None and not ECS_dataframe.empty:
                df = ECS_dataframe.drop_duplicates(subset=['TripEventID']).copy()
                ECS_dataframeshape = df.shape
                ECS_Dataframe_nodup = ECS_helper.remove_duplicates(ECS_dataframe)
                df1 = ECS_Dataframe_nodup.drop_duplicates(subset=['TripEventID']).copy()
                
                ECS_Dataframe_nodupshape = df1.shape
                
                clean_df = ECS_helper.data_preperation(ECS_Dataframe_nodup)
                clean_dfshape = clean_df.shape

            # filename = r'optimized_xgb_model.pkl'

                if clean_df.empty:
                    logging.info(f'No data to Predict. Wait for the next run')
                else:
                    import sklearn
                    logging.info(f'{sklearn.__version__}')
                    import numpy
                    logging.info(f'numpy version: {numpy.__version__}')
                    

                    pred_df = ECS_helper.data_prediction(filename, clean_df)
                    pred_dfshape = pred_df.shape

                    merged_df = pd.merge(pred_df, ECS_dataframe, how='inner', on='TripEventID')
                    score_df, classification_df = ECS_helper.ECS_claculate_importance_score(merged_df)
                    logging.info(f'importance score - {score_df.shape}')
                    jsonResult = score_df.head().to_json(orient='records')
                    final_output = {
                        "header": name,
                        "data": json.loads(jsonResult)
                    }
                    ECS.DA_update_data(score_df,classification_df, name, 'DbServer')
                    return json.dumps(final_output)
                    
            else:
                ECS_dataframeshape =(0,0)
                ECS_Dataframe_nodupshape=(0,0)
                clean_dfshape = (0,0)
                pred_dfshape = (0,0)
                logging.info(f'No data to Predict. Wait for the next run')
            logging.info(f'the data stats are = {ECS_dataframeshape[0]}, {ECS_Dataframe_nodupshape[0]}, {clean_dfshape[0]}, {pred_dfshape[0]}')
        except Exception as e:
            logging.error(f'func_ECS exception as {e}')

    elif servername == 'Db2Server':

        try:

            ECS_dataframe = ECS.ECS_get_data(name, 'Db2Server')
            if ECS_dataframe is not None and not ECS_dataframe.empty:
                df = ECS_dataframe.drop_duplicates(subset=['TripEventID']).copy()
                ECS_dataframeshape = df.shape
                ECS_Dataframe_nodup = ECS_helper.remove_duplicates(ECS_dataframe)
                df1 = ECS_Dataframe_nodup.drop_duplicates(subset=['TripEventID']).copy()
                
                ECS_Dataframe_nodupshape = df1.shape
                
                clean_df = ECS_helper.data_preperation(ECS_Dataframe_nodup)
                clean_dfshape = clean_df.shape

                # filename = r'optimized_xgb_model.pkl'

                if clean_df.empty:
                    logging.info(f'No data to Predict. Wait for the next run')
                else:
                    import sklearn
                    logging.info(f'{sklearn.__version__}')
                    import numpy
                    logging.info(f'numpy version: {numpy.__version__}')
                    

                    pred_df = ECS_helper.data_prediction(filename, clean_df)
                    pred_dfshape = pred_df.shape

                    merged_df = pd.merge(pred_df, ECS_dataframe, how='inner', on='TripEventID')
                    score_df, classification_df = ECS_helper.ECS_claculate_importance_score(merged_df)
                    jsonResult = score_df.head().to_json(orient='records')
                    final_output = {
                        "header": name,
                        "data": json.loads(jsonResult)
                    }
                    ECS.DA_update_data(score_df,classification_df, name, 'Db2Server')
                    return json.dumps(final_output)
                    logging.info(f'importance score - {score_df.shape}')
                    
            else:
                ECS_dataframeshape =(0,0)
                ECS_Dataframe_nodupshape=(0,0)
                clean_dfshape = (0,0)
                pred_dfshape = (0,0)
                logging.info(f'No data to Predict. Wait for the next run')
            logging.info(f'the data stats are = {ECS_dataframeshape[0]}, {ECS_Dataframe_nodupshape[0]}, {clean_dfshape[0]}, {pred_dfshape[0]}')
        except Exception as e:
            logging.error(f'func_ECS exception as {e}')


    else:

        logging.error(f'Company ID not present in our database')

    

