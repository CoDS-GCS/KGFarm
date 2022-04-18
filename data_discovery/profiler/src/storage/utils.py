import json

from past.builtins import xrange


def serialize_rawData(rawData: list):
    for rd in rawData:
        values = rd.get_values()
        chunks = [values[x:x + 1000] for x in xrange(0, len(values), 1000)]
        for chunk in chunks:
            action = {'_index': 'raw_data'}
            data = {'id': rd.get_rid(), 'origin': rd.get_origin(), 'datasetName': rd.get_dataset_name(),
                    'path': rd.get_path(),
                    'tableName': rd.get_table_name(), 'columnName': rd.get_column_name(),
                    'values': json.dumps(chunk)}

            action['_source'] = data
            yield action


def serialize_profiles(profiles: list):
    for p in profiles:
        action = {'_index': 'profiles'}
        data = {'id': p.get_pid(), 'origin': p.get_origin(), 'datasetName': p.get_dataset_name(), 'path': p.get_path(),
                'tableName': p.get_table_name(), 'columnName': p.get_column_name(),
                'dataType': p.get_data_type(), 'totalValuesCount': p.get_total_values(),
                'distinctValuesCount': p.get_distinct_values_count(),
                'missingValuesCount': p.get_missing_values_count(),
                'minValue': p.get_min_value(), 'maxValue': p.get_max_value(), 'avgValue': p.get_mean(),
                'median': p.get_median(), 'iqr': p.get_iqr(), 'minhash': json.dumps(p.get_minhash())}
        action['_source'] = data
        yield action
