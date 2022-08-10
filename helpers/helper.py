import os
import io
import yaml
import stardog
import numpy as np
import pandas as pd
import seaborn as sns
import SPARQLWrapper.Wrapper
from matplotlib import pyplot as plt
from SPARQLWrapper import JSON, SPARQLWrapper


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


# TODO: rename return_type to appropriate variable name to avoid confusion
def execute_query(conn: stardog.Connection, query: str, return_type: str = 'csv'):
    query = PREFIXES + query
    if return_type == 'csv':
        result = conn.select(query, content_type='text/csv')
        return pd.read_csv(io.BytesIO(bytes(result)))
    elif return_type == 'json':
        result = conn.select(query)
        return result['results']['bindings']
    elif return_type == 'ask':
        result = conn.select(query)
        return result['boolean']
    elif return_type == 'update':
        result = conn.update(query)
        return result
    else:
        error = return_type + ' not supported!'
        raise ValueError(error)


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


def plot_comparison(conventional_approach: dict, kgfarm_approach: dict):
    plt.rcParams['figure.dpi'] = 300
    sns.set_style("dark")
    scores = {}
    for classifier in conventional_approach.keys():
        scores[classifier] = [conventional_approach.get(classifier), kgfarm_approach.get(classifier)]

    labels = ["conventional pipeline", "using KGFarm"]
    x = np.arange(len(labels))
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(7, 6))

    classifier = 'Random forest classifier'
    color = 'darkgreen'
    rects1 = ax.bar(x - width, scores.get(classifier), color=color, edgecolor=color, width=width, label=classifier)
    plt.axhline(y=scores.get(classifier)[1], color=color, linestyle="--")
    classifier = 'Gradient boosting classifier'
    color = 'mediumseagreen'
    rects2 = ax.bar(x, scores.get(classifier), color=color, width=width, edgecolor=color, label=classifier)
    plt.axhline(y=scores.get(classifier)[1], color=color, linestyle="--")
    classifier = 'Naive bayes classifier'
    color = 'lightgreen'
    rects3 = ax.bar(x + width, scores.get(classifier), color=color, width=width, edgecolor=color, label=classifier)
    plt.axhline(y=scores.get(classifier)[1], color=color, linestyle="--")

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h, float(h), ha='center', verticalalignment='bottom',
                    fontsize=15)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_ylabel('F1-score', fontsize=15)
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(("conventional ML pipeline", "using KGFarm"), fontsize=15)
    ax.legend(fontsize=10, bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1.35)
    plt.show()

