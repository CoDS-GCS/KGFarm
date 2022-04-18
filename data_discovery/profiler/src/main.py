from orchestration.orchestrator import Orchestrator
from helpers.helper import *
import time


def start_profiling():
    start_time = time.time()
    orchestrator = Orchestrator()
    orchestrator.create_tables('../config/config.yml')
    print('\nProcessing tables')
    orchestrator.process_tables(7)
    print("Profiled in ", time_taken(start_time, time.time()))


refresh_elasticsearch()
setup_config(path='../../../helpers/sample_data/parquet/', datasource='kaggle', datatype='parquet')
start_profiling()
