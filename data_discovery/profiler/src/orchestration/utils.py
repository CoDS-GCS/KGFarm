import os
import queue
from data.tables.parquet_table import *
from data.tables.csv_table import *
from data.utils.file_type import FileType


def get_file_type(filename: str) -> FileType:
    if filename.endswith('.csv'):
        return FileType.CSV
    elif filename.endswith('.parquet'):
        return FileType.PARQUET


def create_sources_from_datasets(datasets: list, tables: queue):
    for dataset in datasets:
        for filename in os.listdir(dataset['path']):
            if get_file_type(filename) == FileType.CSV:
                csvTable = CSVTable(filename, dataset['name'], dataset['path'], dataset['origin'], 'csv')
                tables.put(csvTable)
            elif get_file_type(filename) == FileType.PARQUET:
                parquetTable = PARQUETTable(filename, dataset['name'], dataset['path'], dataset['origin'], 'parquet')
                tables.put(parquetTable)
