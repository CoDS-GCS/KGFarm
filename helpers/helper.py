import os
import io
import yaml
import stardog
import pandas as pd
import SPARQLWrapper.Wrapper
from SPARQLWrapper import JSON, SPARQLWrapper

PREFIXES = """
    PREFIX kgfarm:      <http://kgfarm.com/ontology/>
    PREFIX kglids:      <http://kglids.org/ontology/>
    PREFIX data:        <http://kglids.org/ontology/data/>
    PREFIX schema:      <http://schema.org/>
    PREFIX rdf:         <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs:        <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX pipeline:    <http://kglids.org/ontology/pipeline/>
     """


def refresh_elasticsearch():
    print('Refreshing elasticsearch!')
    status = os.system("curl -X DELETE 'http://localhost:9200/_all'")


def time_taken(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def setup_config(path: str, datasource: str, datatype: str):
    def write_to_config(config_to_dump: dict):
        data_source = {'datasource': datasource}
        with open('../config/config.yml', 'w') as file:
            yaml.dump(data_source, file)
        with open('../config/config.yml', 'a') as file:
            yaml.dump(config_to_dump, file)

    os.system('find {} -name "*.DS_Store" -type f -delete'.format(path))
    temp = []
    table_count = dataset_count = 0
    for dataset in os.listdir(path):
        dataset_name = dataset
        dataset = path + dataset_name
        temp.append({'name': dataset_name, 'path': dataset, 'type': datatype, 'origin': 'mock'})
        dataset_count = dataset_count + 1
        for table in os.listdir(dataset):
            if table.endswith('.' + datatype):
                table_count = table_count + 1

    config = {'datasets': temp}
    print('\n\nNow Profiling: \n• # Dataset(s): {}\n• # Table(s)  : {}'.format(dataset_count, table_count))
    write_to_config(config)


def upload_glac(path: str, namespace: str, port=9999):
    command = "curl -D- -H 'Content-Type: application/x-turtle-RDR' --upload-file {} -X POST 'http://localhost:{}/blazegraph/namespace/{}/sparql'".format(
        path, port, namespace)
    print('\n\n• Uploading Glac to blazegraph!\n')
    os.system(command)


def drop_glac(namespace: str, port=9999):
    command = "curl --get -X DELETE 'http://localhost:{}/blazegraph/namespace/{}/sparql'".format(port, namespace)
    print('\n• Dropping existing graph hosted on blazegraph!\n')
    os.system(command)


def connect_to_blazegraph(port, namespace, show_status: bool):
    endpoint = 'http://localhost:{}/blazegraph/namespace/'.format(port) + namespace + '/sparql'
    if show_status:
        print('connected to {}'.format(endpoint))
    return SPARQLWrapper(endpoint)


def connect_to_stardog(port, db: str, show_status: bool):
    connection_details = {
        'endpoint': 'http://localhost:{}'.format(str(port)),
        'username': 'admin',
        'password': 'admin'
    }
    conn = stardog.Connection(db, **connection_details)
    if show_status:
        print('Connected to Stardog!\nAccess the Stardog UI at: https://cloud.stardog.com/')
    return conn


def execute_query_blazegraph(sparql: SPARQLWrapper, query: str):
    sparql.setQuery(PREFIXES + query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def execute_query(conn: stardog.Connection, query: str, return_type: str = 'csv'):
    query = PREFIXES + query
    if return_type == 'csv':
        result = conn.select(query, content_type='text/csv')
        return pd.read_csv(io.BytesIO(result))
    elif return_type == 'json':
        result = conn.select(query)
        return result['results']['bindings']
    else:
        raise ValueError(return_type, ' not supported')


def display_query(query: str):
    query = PREFIXES + query
    print(query)


def convert_dict_to_dataframe(key, d: dict):
    df = {}
    first_col = list(d.keys())
    columns = list(d[next(iter(d))].keys())
    for i in range(len(columns)):
        for k, v in d.items():
            if columns[i] in df:
                val = df.get(columns[i])
                val.append(v[columns[i]])
                df[columns[i]] = val
            else:
                df[columns[i]] = [v[columns[i]]]

    df = pd.DataFrame.from_dict(df)
    df.insert(loc=0, column=key, value=first_col)
    return df
