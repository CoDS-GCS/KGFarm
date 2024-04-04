import io
import os
import urllib.parse
import SPARQLWrapper.Wrapper
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from urllib.parse import quote_plus
from SPARQLWrapper import JSON, SPARQLWrapper, CSV
from matplotlib import pyplot as plt

PREFIXES = """
    PREFIX kgfarm:      <http://kgfarm.com/ontology/>
    PREFIX featureView: <http://kgfarm.com/ontology/featureView/>
    PREFIX entity:      <http://kgfarm.com/ontology/entity/>
    PREFIX kglids:      <http://kglids.org/ontology/>
    PREFIX data:        <http://kglids.org/ontology/data/>
    PREFIX schema:      <http://schema.org/>
    PREFIX rdf:         <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs:        <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX pipeline:    <http://kglids.org/ontology/pipeline/>
    PREFIX lib:         <http://kglids.org/resource/library/> 
    """


def refresh_elasticsearch():
    print('Refreshing elasticsearch!')
    _ = os.system("curl -X DELETE 'http://localhost:9200/_all'")


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


# def connect_to_stardog(port, db: str, show_status: bool):
#     connection_details = {
#         'endpoint': 'http://localhost:{}'.format(str(port)),
#         'username': 'admin',
#         'password': 'admin'
#     }
#     conn = stardog.Connection(db, **connection_details)
#     if show_status:
#         print('Connected to Stardog!\nAccess the Stardog UI at: https://cloud.stardog.com/')
#     return conn

def connect_to_graphdb(endpoint, graphdb_repo):
    graphdb = SPARQLWrapper(f'{endpoint}/repositories/{graphdb_repo}')
    return graphdb

def execute_query_blazegraph(sparql: SPARQLWrapper, query: str):
    sparql.setQuery(PREFIXES + query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


# TODO: rename return_type to appropriate variable name to avoid confusion
# def execute_query(graphdb_conn: SPARQLWrapper, query, return_type='csv'):
#     graphdb_conn.setQuery(query)
#     if return_type == 'csv':
#         graphdb_conn.setReturnFormat(CSV)
#         results = graphdb_conn.queryAndConvert()
#         return pd.read_csv(io.BytesIO(results))
#     elif return_type == 'json':
#         graphdb_conn.setReturnFormat(JSON)
#         results = graphdb_conn.queryAndConvert()
#         return results['results']['bindings']
#     else:
#         raise ValueError(return_type, ' not supported')
def execute_query(sparql, query):
    sparql.setQuery(PREFIXES + query)
    sparql.setReturnFormat(CSV)
    results = sparql.query().convert()
    data_str = results.decode('utf-8')
    df = pd.read_csv(io.StringIO(data_str))
    return df
# def execute_query(conn: stardog.Connection, query: str, return_type: str = 'csv', timeout: int = 0):
#     query = PREFIXES + query
#     if return_type == 'csv':
#         result = conn.select(query, content_type='text/csv', timeout=timeout)
#         return pd.read_csv(io.BytesIO(bytes(result)))
#     elif return_type == 'json':
#         result = conn.select(query)
#         return result['results']['bindings']
#     elif return_type == 'ask':
#         result = conn.select(query)
#         return result['boolean']
#     elif return_type == 'update':
#         result = conn.update(query)
#         return result
#     else:
#         error = return_type + ' not supported!'
#         raise ValueError(error)


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


def plot_scores(conventional_approach: dict):
    plt.rcParams['figure.dpi'] = 120
    sns.set_style("dark")

    x = list(classifier.replace('classifier', '') for classifier in conventional_approach.keys())
    y = list(conventional_approach.values())

    fig, ax = plt.subplots(figsize=(7, 3))
    width = 0.38
    ind = np.arange(len(y))
    color = 'tab:green'
    ax.barh(ind, y, width, color=color, edgecolor=color)

    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(x, minor=False)
    ax.invert_yaxis()
    plt.xlabel('F1-score')
    plt.grid(color='lightgray', linestyle='dashed', axis='x')
    plt.show()


def plot_comparison(baseline_approach: dict, kgfarm_approach: dict):
    baseline_approach = {key: round(value, 2) for key, value in baseline_approach.items()}
    kgfarm_approach = {key: round(value, 2) for key, value in kgfarm_approach.items()}

    # Extract keys and values from dictionaries
    keys = list(baseline_approach.keys())
    values1 = list(baseline_approach.values())
    values2 = list(kgfarm_approach.values())

    bar_width = 0.35

    r1 = range(len(keys))
    r2 = [x + bar_width for x in r1]
    offset = 0.7
    values1_adjusted = [v - 0.7 for v in values1]
    values2_adjusted = [v - 0.7 for v in values2]
    plt.bar(r1, values1_adjusted, color='#B3B3B3', width=bar_width, edgecolor='grey', label='baseline_approach',
            bottom=offset)
    plt.bar(r2, values2_adjusted, color='#1AA260', width=bar_width, edgecolor='grey', label='kgfarm_approach',
            bottom=offset)
    plt.xlabel('Models', fontweight='bold')
    plt.xticks([r + bar_width / 2 for r in range(len(keys))], keys)
    plt.ylabel('R2 Score', fontweight='bold')

    for i, v in enumerate(values1):
        plt.text(i, v, round(v, 2), ha='center', va='bottom', fontsize=14)
    for i, v in enumerate(values2):
        plt.text(i + bar_width, v, round(v, 2), ha='center', va='bottom', fontsize=14)
    plt.ylim(0.7, 0.95)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def generate_column_id(profile_path: str, column_name: str):
    def url_encode(string):
        return urllib.parse.quote(str(string), safe='')  # safe parameter is important.

    profile_path = profile_path.split('/')
    table_name = profile_path[-1]
    dataset_name = profile_path[-3]

    return f"http://kglids.org/resource/{url_encode('kaggle')}/" \
           f"{url_encode(dataset_name)}/dataResource/{url_encode(table_name)}/" \
           f"{url_encode(column_name)}"


def generate_table_id(profile_path: str):
    profile_path = profile_path.split('/')
    table_name = profile_path[-1]
    dataset_name = profile_path[-3]
    table_id = f'http://kglids.org/resource/kaggle/{quote_plus(dataset_name)}/dataResource/{quote_plus(table_name)}'
    return table_id
