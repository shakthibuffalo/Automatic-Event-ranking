import logging

from azure.functions import TimerRequest
from datetime import datetime, timezone
from azure.durable_functions import DurableOrchestrationClient
import AER_constants


async def main(mytimer: TimerRequest, starter: str) -> None:
    utc_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    logging.info('Python timer trigger function started at %s', utc_timestamp)

    if mytimer.past_due:
        logging.info('The timer is past due!')

    client = DurableOrchestrationClient(starter)
    namelist = AER_constants.namelist3

    instance_id = await client.start_new("FunctionOrchestrator", None, namelist)

    logging.info(f"Started orchestration with ID = '{instance_id}'.")

