import pandas as pd
from SPARQLWrapper import JSON
from helpers.helper import execute_query_blazegraph


PREFIXES = """
    prefix lac:     <http://www.example.com/lac#>
    prefix schema:  <http://schema.org/>
    prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    prefix dct:     <http://purl.org/dc/terms/>
     """


def predict_entities_and_feature_views(config, thresh, show_query):
    query = PREFIXES + """
       SELECT DISTINCT ?table_name ?column1_name ?column_type ?source_table_path ?table2_name ?column2_name ?certainty
       WHERE
       {
           <<?column_id lac:pkfk ?column_id_x>> lac:certainty	?certainty	.
           FILTER (?certainty >= %s)                                    .
           ?column_id    schema:name	    ?column1_name	.
           ?column_id    schema:type     ?column_type     .
           ?column_id_x	schema:name	    ?column2_name	.
           ?column_id    dct:isPartOf	?table_id	.
           ?table_id     lac:path       ?source_table_path  .
           ?column_id_x	dct:isPartOf	?table2_id	.
           ?table_id	schema:name		?table_name	.
           ?table2_id	schema:name		?table2_name	.
       }""" % thresh

    if show_query:
        print(query)

    results = execute_query_blazegraph(config, query)
    bindings = results["results"]["bindings"]
    if not bindings:
        return bindings
    for result in bindings:
        yield {'source_table': result['table_name']['value'],
               'source_column': result['column1_name']['value'],
               'source_column_type': result['column_type']['value'],
               'source_table_path': result['source_table_path']['value'],
               'target_table': result['table2_name']['value'],
               'target_column': result['column2_name']['value'],
               'confidence_score': result['certainty']['value']}
