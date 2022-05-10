from helpers.helper import execute_query_blazegraph, execute_query, display_query
import numpy as np

PREFIXES = """
    PREFIX lac:     <http://www.example.com/lac#>
    PREFIX schema:  <http://schema.org/>
    PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dct:     <http://purl.org/dc/terms/>
     """


def get_columns(config, table, dataset):
    query = PREFIXES + """
    SELECT DISTINCT ?Column	
    WHERE
    {
        ?Table_id			schema:name		"%s"				                    .
        ?Dataset_id			schema:name		"%s"				                    .
        ?Table_id			dct:isPartOf	?Dataset_id								.
        ?Column_id			dct:isPartOf	?Table_id								;
                            schema:name		?Column									.
    }
    """ % (table, dataset)

    return execute_query(config, query)['Column'].tolist()


def predict_entities(config, show_query):
    query = """
    SELECT DISTINCT ?Entity ?Entity_data_type ?File_source ?File_source_path ?Dataset
    WHERE
    {
      <<?Entity_id 	lac:pkfk ?column_id_1>> lac:certainty ?score	.
      ?Entity_id	schema:totalVCount		?Total_values			;
      				schema:distinctVCount	?Distinct_values		;
                    schema:name				?Entity					;
      				schema:type				?Entity_data_type		.
      
      FILTER(?Total_values = ?Distinct_values)					    .
      
      ?Entity_id	dct:isPartOf			?Table_id				.
      ?Table_id     schema:name             ?File_source            ;
                    lac:path				?File_source_path       ;
                    dct:isPartOf            ?Dataset_id             .
      
      ?Dataset_id   schema:name             ?Dataset               .
    }"""
    if show_query:
        display_query(query)

    entities = execute_query(config, query). \
        replace(['N', 'T'], ['INT64', 'STRING'])
    return entities


def get_enrichment_tables(config, show_query):
    query = """
    SELECT DISTINCT ?Table ?Entity ?Path_to_table ?Dataset
    WHERE
    {
      <<?Entity_id 			lac:pkfk ?column_id_1>> lac:certainty ?Score	.
      ?Entity_id			schema:totalVCount		?Total_values			;
      						schema:distinctVCount	?Distinct_values		;
                    		schema:name				?Entity					;
      						schema:type				?Entity_data_type		;
                    		dct:isPartOf			?Table_id				.
      
      FILTER(?Total_values = ?Distinct_values)					    		.
      	
      ?Table_id     		schema:name             ?File_source            ;
      						lac:path				?File_source_path       .
      
      ?column_id_1			dct:isPartOf			?Joinable_table_id		.
      ?Joinable_table_id	schema:name				?Table      			;
                            lac:path				?Path_to_table			;
                            dct:isPartOf            ?Dataset_id             .
                
      ?Dataset_id           schema:name             ?Dataset                .            
      FILTER(?Table_id != ?Joinable_table_id)					    		.
    } ORDER BY DESC (?Entity)
    """
    if show_query:
        display_query(query)

    return execute_query(config, query)


# TODO: add support for all predicted joinable tables for kgfarm.predict_features()
def predict_features(config, table, dataset, show_query):
    query = PREFIXES + """
    SELECT DISTINCT ?Source_table ?Joinable_table (?Column_id as ?Join_key_id)
    WHERE
    {
        ?Table_id			schema:name 	"%s"					                .
        ?Dataset_id			schema:name		"%s"				                    .
        ?Table_id   		dct:isPartOf 	?Dataset_id								.
        ?Table_id			schema:name		?Source_table							.
        ?Column_id			dct:isPartOf	?Table_id								.
        <<?Column_id		lac:pkfk		?Column_id_2>> lac:certainty	?Score	.
        ?Column_id_2		dct:isPartOf	?Joinable_table_id						.
        ?Joinable_table_id	schema:name		?Joinable_table							
    }""" % (table, dataset)
    if show_query:
        print(query)
    df = execute_query(config, query)
    joinable_table = df['Joinable_table'].iloc[0]
    get_columns(config, joinable_table, dataset)
    columns_to_join = [i for i in get_columns(config, joinable_table, dataset) if
                       i not in get_columns(config, table, dataset)]
    return joinable_table, columns_to_join
