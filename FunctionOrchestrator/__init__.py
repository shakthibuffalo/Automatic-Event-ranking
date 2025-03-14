# This function is not intended to be invoked directly. Instead it will be
# triggered by an HTTP starter function.
# Before running this sample, please:
# - create a Durable activity function (default name is "Hello")
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import json

from azure.durable_functions import DurableOrchestrationContext, Orchestrator


def orchestrator_function(context: DurableOrchestrationContext):
    names = context.get_input()
    logging.info(f'names from hpptstarter are = {names}')
    logging.info(f'logging a step before starting func_whatever')

    task_event_type_1 = [context.call_activity('func_overspeed', name) for name in names]

    task_event_type_2 = [context.call_activity('func_nonsevereOSE', name) for name in names]

    task_event_type_3 = [context.call_activity('func_DA', name) for name in names]

    task_event_type_4 = [context.call_activity('func_ECS', name) for name in names]

    task_event_type_5 = [context.call_activity('func_CMB_FCW_EB', name) for name in names]

    task_event_type_6 = [context.call_activity('func_ESPRSP', name) for name in names]

    # parallel_task = [context.call_activity('func_CMB_FCW_EB', name) for name in names]

    # results = yield context.task_all(parallel_task)

    # results = yield context.task_all([
    #     context.task_all(task_event_type_1),
    #     context.task_all(task_event_type_2)   
    # ])

    results = yield context.task_all([
        context.task_all(task_event_type_1),
        context.task_all(task_event_type_2),
        context.task_all(task_event_type_3),
        context.task_all(task_event_type_4),
        context.task_all(task_event_type_5),
        context.task_all(task_event_type_6)
    ])

    return results

main = Orchestrator.create(orchestrator_function)