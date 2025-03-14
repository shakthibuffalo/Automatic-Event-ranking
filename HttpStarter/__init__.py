# This function an HTTP starter function for Durable Functions.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable activity function (default name is "Hello")
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt
 
import logging

from AER_helper import *
import AER_constants
from azure.functions import HttpRequest, HttpResponse
from azure.durable_functions import DurableOrchestrationClient


async def main(req: HttpRequest, starter: str) -> HttpResponse:
    client = DurableOrchestrationClient(starter)

    AER_helper = AER_helperfunction()
    # namelist = AER_helper.DA_getautoeventclass_CID()

    # namelist = [[49, 'SafetyDirect2_shard65', 65], [83, 'SafetyDirect2_shard63', 63], [114, 'SafetyDirect2_shard24', 24], [182, 'SafetyDirect2_shard314', 314], [234, 'SafetyDirect2_shard421', 421]]

    namelist = AER_constants.namelist3
    instance_id = await client.start_new(req.route_params["functionName"], None, namelist)

    logging.info(f"Started orchestration with ID = '{instance_id}'.")

    return client.create_check_status_response(req, instance_id)