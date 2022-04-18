import os
import zlib


def generate_id(dataset_name: str, table_name: str, column_name: str):
    return zlib.crc32(bytes(dataset_name + table_name + column_name, 'utf-8'))


