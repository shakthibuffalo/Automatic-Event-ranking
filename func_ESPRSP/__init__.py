# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging

from RSPESP_helper import *

import AER_constants
from AER_helper import *


def main(name: str) -> str:

    ESPRSP_classifier = ESPRSP_Classification()
    ESPRSP_dataret = RSP_ESP()
    ESPRSP_helper = data_preperation()
    AER_helper = AER_helperfunction()

    filename = r'ESPRSPbest_random_forest_model1.pkl'

    servername = AER_helper.AER_getserveripmapping(name, AER_constants.DataSource_mapping_dict)
    logging.info(f'the server name is {servername}')

    if servername == 'DbServer':
        
        try:
            ESPRSP_Dataframe = ESPRSP_classifier.RSPESP_get_data(name, 'DbServer')
            if ESPRSP_Dataframe is None or ESPRSP_Dataframe.empty:
                None
            else:
                import sklearn
                logging.info(f'{sklearn.__version__}')
                import numpy
                logging.info(f'numpy version: {numpy.__version__}')

                def calc_noofAbst(group):
                    group = group.sort_values(by='ppetimestamp')
                    count = 0
                    inseq = False
                    
                    for num in group['ppeAbs']:
                        if num == 1:
                            if not inseq:
                                inseq = True
                                count += 1
                        else:
                            inseq = False
                    logging.info(f'count - {count}')
                    return count
                ESPRSP_Dataframe['abscount'] = ESPRSP_Dataframe.groupby('TripEventID', group_keys=False).apply(lambda group: calc_noofAbst(group))
                
                # ESPRSP_Dataframe['abscount'] = ESPRSP_Dataframe.groupby('TripEventID', 
                #                                                         group_keys=False).apply(lambda group: pd.Series(calc_noofAbst(group), 
                #                                                                                                         index=group.index))
             
                # ESPRSP_Dataframe['abscount'] = ESPRSP_Dataframe.groupby('TripEventID', group_keys=False).apply(
                #         lambda group: calc_noofAbst(group)
                    # )
                # ESPRSP_Dataframe['abscount'] = ESPRSP_Dataframe.groupby('TripEventID')['ppeAbs'].transform(calc_noofAbst)

                # ESPRSP_Dataframe['abscount'] = ESPRSP_Dataframe.groupby('TripEventID').apply(lambda group: pd.Series(calc_noofAbst(group), 
                # index=group.index)).reset_index(level=0, 
                # drop=True)



                ESPRSP_Dataframe['tetimestamp'] = pd.to_datetime(ESPRSP_Dataframe['tetimestamp'])
                ESPRSP_Dataframe['ppetimestamp'] = pd.to_datetime(ESPRSP_Dataframe['ppetimestamp'])

                ESPRSP_Dataframe = ESPRSP_Dataframe.sort_values(by=['TripEventID','ppetimestamp'])
                ESPRSP_Dataframe['isSevere'] = 0
                
                ESPRSP_Dataframe_cleaned = ESPRSP_helper.feature_generation(ESPRSP_Dataframe)

                ESPRSP_Dataframe_cleaned['abscountlast'] = ESPRSP_Dataframe_cleaned['abscountlast'].fillna(0)

                ESPRSP_Datafreane_predicted, ESPRSP_Datadrame_classofied = ESPRSP_helper.data_prediction(filename, ESPRSP_Dataframe_cleaned)

                logging.info(f'predicted dataframe {ESPRSP_Datafreane_predicted.shape} and classified dataframe {ESPRSP_Datadrame_classofied.shape}')

                jsonResult = ESPRSP_Datafreane_predicted.head().to_json(orient='records')
                final_output = {
                        "header": name,
                        "data": json.loads(jsonResult)
                    }
                ESPRSP_classifier.RSPESP_update_data(ESPRSP_Datafreane_predicted, ESPRSP_Datadrame_classofied, name, 'DbServer')
                return json.dumps(final_output)

                
        except Exception as e:
            logging.error(f'func_esprsp exception for {name} as {e}')

    elif servername == 'Db2Server':
        
        try:
            ESPRSP_Dataframe = ESPRSP_classifier.RSPESP_get_data(name, 'Db2Server')
            if ESPRSP_Dataframe is None or ESPRSP_Dataframe.empty:
                None
            else:
                import sklearn
                logging.info(f'{sklearn.__version__}')
                import numpy
                logging.info(f'numpy version: {numpy.__version__}')

                def calc_noofAbst(group):
                    group = group.sort_values(by='ppetimestamp')
                    count = 0
                    inseq = False
                    
                    for num in group['ppeAbs']:
                        if num == 1:
                            if not inseq:
                                inseq = True
                                count += 1
                        else:
                            inseq = False
                    logging.info(f'count - {count}')
                    return count
                
                # ESPRSP_Dataframe['abscount'] = ESPRSP_Dataframe.groupby('TripEventID', 
                #                                                         group_keys=False).apply(lambda group: pd.Series(calc_noofAbst(group), 
                #                                                                                                         index=group.index))
                
                ESPRSP_Dataframe['abscount'] = ESPRSP_Dataframe.groupby('TripEventID', group_keys=False).apply(
                        lambda group: calc_noofAbst(group)
                    )
                # ESPRSP_Dataframe['abscount'] = ESPRSP_Dataframe.groupby('TripEventID')['ppeAbs'].transform(calc_noofAbst)

                # ESPRSP_Dataframe['abscount'] = ESPRSP_Dataframe.groupby('TripEventID').apply(lambda group: pd.Series(calc_noofAbst(group), 
                # index=group.index)).reset_index(level=0, 
                # drop=True)

                ESPRSP_Dataframe['tetimestamp'] = pd.to_datetime(ESPRSP_Dataframe['tetimestamp'])
                ESPRSP_Dataframe['ppetimestamp'] = pd.to_datetime(ESPRSP_Dataframe['ppetimestamp'])

                ESPRSP_Dataframe = ESPRSP_Dataframe.sort_values(by=['TripEventID','ppetimestamp'])
                ESPRSP_Dataframe['isSevere'] = 0
                
                ESPRSP_Dataframe_cleaned = ESPRSP_helper.feature_generation(ESPRSP_Dataframe)

                ESPRSP_Datafreane_predicted, ESPRSP_Datadrame_classofied = ESPRSP_helper.data_prediction(filename, ESPRSP_Dataframe_cleaned)

                logging.info(f'predicted dataframe {ESPRSP_Datafreane_predicted.shape} and classified dataframe {ESPRSP_Datadrame_classofied.shape}')

                jsonResult = ESPRSP_Datafreane_predicted.head().to_json(orient='records')
                final_output = {
                        "header": name,
                        "data": json.loads(jsonResult)
                    }
                ESPRSP_classifier.RSPESP_update_data(ESPRSP_Datafreane_predicted, ESPRSP_Datadrame_classofied, name, 'Db2Server')
                return json.dumps(final_output)


                
        except Exception as e:
            logging.error(f'func_esprsp exception for {name} as {e}')
    else:
        logging.info('no server')


    # return f"Hello {name}!"
