# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import testfile2
from AER_helper import *
import AER_constants


def main(name: str) -> str:
    logging.info(f'Processing name {name}')

    DA = testfile2.Distance_alert_classification()
    DA_helper = testfile2.Distance_alert_helper()
    AER_helper = AER_helperfunction()

    servername = AER_helper.AER_getserveripmapping(name, AER_constants.DataSource_mapping_dict)

    if servername == 'DbServer':
        try:

            DA_Dataframe = DA.DA_get_data(name, 'DbServer')
            if DA_Dataframe is not None and not DA_Dataframe.empty:
                DA_Importance_Scores_Dataframe, DA_classification_Dataframe = DA_helper.calculate_scores(DA_Dataframe)

                jsonResult = DA_Importance_Scores_Dataframe.head().to_json(orient='records')
                final_output = {
                            "header": name,
                            "data": json.loads(jsonResult)
                        }
                
                DA.DA_update_data(DA_Importance_Scores_Dataframe,DA_classification_Dataframe, name, 'DbServer')
                return json.dumps(final_output)
            else:
                return f'No output was returned in this company - {name}'

            
            
            
        except Exception as e:
            logging.error(f'func_DA exception as {e}')

    elif servername == 'Db2Server':

        try:
            DA_Dataframe = DA.DA_get_data(name, 'Db2Server')

            if DA_Dataframe is not None and not DA_Dataframe.empty:

                DA_Importance_Scores_Dataframe, DA_classification_Dataframe = DA_helper.calculate_scores(DA_Dataframe)

                jsonResult = DA_Importance_Scores_Dataframe.head().to_json(orient='records')
                final_output = {
                            "header": name,
                            "data": json.loads(jsonResult)
                        }
                
                DA.DA_update_data(DA_Importance_Scores_Dataframe,DA_classification_Dataframe, name, 'Db2Server')
                return json.dumps(final_output)
            else:
                return f'No output was returned in this company - {name}'

            

        except Exception as e:
            logging.error(f'func_DA exception as {e}')

    # if DA_Importance_Scores_Dataframe is not None and not DA_Importance_Scores_Dataframe.empty:
    #     jsonResult = DA_Importance_Scores_Dataframe.head().to_json(orient='records')
    #     final_output = {
    #                 "header": name,
    #                 "data": json.loads(jsonResult)
    #             }
    #     return json.dumps(final_output)
    # else:
    #     return f'No output was returned in this company - {name} - {DA_Importance_Scores_Dataframe}'
    
    # return f"Hello {name}!"
