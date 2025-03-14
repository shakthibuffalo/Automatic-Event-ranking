# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import OverSPeed_mainfile
from AER_helper import *
import AER_constants


def main(name: str) -> str:

    OSE_Classification = OverSPeed_mainfile.Overspeed_Event_Classification()
    AER_helper = AER_helperfunction()

    servername = AER_helper.AER_getserveripmapping(name, AER_constants.DataSource_mapping_dict)
    OSE_Classification.set_initial_parameters(name)

    if servername == 'DbServer':
        try:
            OSE_Dataframe = OSE_Classification.ose_get_data('DbServer')
            if OSE_Dataframe is not None and not OSE_Dataframe.empty:
                df = OSE_Dataframe.drop_duplicates(subset=['TripEventID']).copy()

                OSE_DataframeSHAPE = df.shape

                OSE_Dataframe_nodup = OSE_Classification.remove_duplicates(OSE_Dataframe)
                df1 = OSE_Dataframe_nodup.drop_duplicates(subset=['TripEventID']).copy()
        
                OSE_Dataframe_nodupshape = df1.shape

                OSE_Cleaned_Dataframe = OSE_Classification.ose_prepare_data(OSE_Dataframe)

                if OSE_Cleaned_Dataframe is not None and not OSE_Cleaned_Dataframe.empty:
                    df2 = OSE_Cleaned_Dataframe.drop_duplicates(subset=['TripEventID']).copy()
        
                    OSE_Cleaned_Dataframeshape = df2.shape
                    OSE_Importance_Scores_Dataframe, OSE_Event_classification_Dataframe = OSE_Classification.ose_calculate_importance_score(OSE_Cleaned_Dataframe)
                    OSE_Importance_Scores_Dataframeshape = OSE_Importance_Scores_Dataframe.shape
                    OSE_Classification.ose_update_data(OSE_Importance_Scores_Dataframe, OSE_Event_classification_Dataframe, 'DbServer')
                    jsonResult = OSE_Importance_Scores_Dataframe.head().to_json(orient='records')
                    final_output = {
                        "header": name,
                        "data": json.loads(jsonResult)
                    }
                    return json.dumps(final_output)
                else:
                    OSE_Cleaned_Dataframeshape = (0,0)
                    OSE_Importance_Scores_Dataframeshape = (0,0)
            else:
                OSE_DataframeSHAPE = (0,0)
                OSE_Dataframe_nodupshape = (0,0)
                OSE_Cleaned_Dataframeshape = (0,0)
                OSE_Importance_Scores_Dataframeshape = (0,0)
                
                
            logging.info(f'the data stats are - {OSE_DataframeSHAPE[0]}, {OSE_Dataframe_nodupshape[0]}, {OSE_Cleaned_Dataframeshape[0]}, {OSE_Importance_Scores_Dataframeshape[0]}')

        except Exception as err:
            logging.error(f'OverSpeed function stopped due to error {err}')
    
    elif servername == 'Db2Server':
        try:
            OSE_Dataframe = OSE_Classification.ose_get_data('Db2Server')
            if OSE_Dataframe is not None and not OSE_Dataframe.empty:
                df = OSE_Dataframe.drop_duplicates(subset=['TripEventID']).copy()
                
                OSE_DataframeSHAPE = df.shape

                OSE_Dataframe_nodup = OSE_Classification.remove_duplicates(OSE_Dataframe)
                df1 = OSE_Dataframe_nodup.drop_duplicates(subset=['TripEventID']).copy()
        
                OSE_Dataframe_nodupshape = df1.shape

                OSE_Cleaned_Dataframe = OSE_Classification.ose_prepare_data(OSE_Dataframe)
                

                if OSE_Cleaned_Dataframe is not None and not OSE_Cleaned_Dataframe.empty:
                    df2 = OSE_Cleaned_Dataframe.drop_duplicates(subset=['TripEventID']).copy()
        
                    OSE_Cleaned_Dataframeshape = df2.shape
                
                    OSE_Importance_Scores_Dataframe, OSE_Event_classification_Dataframe = OSE_Classification.ose_calculate_importance_score(OSE_Cleaned_Dataframe)
                    OSE_Importance_Scores_Dataframeshape = OSE_Importance_Scores_Dataframe.shape
                    OSE_Classification.ose_update_data(OSE_Importance_Scores_Dataframe, OSE_Event_classification_Dataframe, 'Db2Server')
                    jsonResult = OSE_Importance_Scores_Dataframe.head().to_json(orient='records')
                    final_output = {
                        "header": name,
                        "data": json.loads(jsonResult)
                    }
                    return json.dumps(final_output)
                else:
                    OSE_Cleaned_Dataframeshape = (0,0)
                    OSE_Importance_Scores_Dataframeshape = (0,0)               
            else:
                OSE_DataframeSHAPE = (0,0)
                OSE_Dataframe_nodupshape = (0,0)
                OSE_Cleaned_Dataframeshape = (0,0)
                OSE_Importance_Scores_Dataframeshape = (0,0)
                

            logging.info(f'the data stats are - {OSE_DataframeSHAPE[0]}, {OSE_Dataframe_nodupshape[0]}, {OSE_Cleaned_Dataframeshape[0]}, {OSE_Importance_Scores_Dataframeshape[0]}')
                
        except Exception as err:
            logging.error(f'OverSpeed function stopped due to error {err}')
            
    else:
        logging.error(f'Company ID not present in our database')

    
        
            

    # return f"Hello {name}!"
