from helpers.helper import execute_query_blazegraph, execute_query

PREFIXES = """
    PREFIX lac:     <http://www.example.com/lac#>
    PREFIX schema:  <http://schema.org/>
    PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dct:     <http://purl.org/dc/terms/>
     """


def predict_entities_and_feature_views(config, thresh, show_query):
    query = PREFIXES + """
    SELECT DISTINCT ?Source_table ?Source_column ?Source_column_type ?Source_table_path ?Target_table ?Target_column ?Certainty_score
    WHERE
    {
      <<?column_id lac:pkfk ?column_id_2>> lac:certainty	?Certainty_score	.
      FILTER (?Certainty_score >= %s)                        					.
      ?column_id    schema:name	    ?Source_column								.
      ?column_id    schema:type     ?Source_column_type     					.
      ?column_id_2	schema:name	    ?Target_column								.
      ?column_id    dct:isPartOf	?table_id									.
      ?table_id     lac:path       	?Source_table_path  						.
      ?column_id_2	dct:isPartOf	?table_id_2									.
      ?table_id		schema:name		?Source_table								.
      ?table_id_2	schema:name		?Target_table								.
    }""" % thresh

    if show_query:
        print(query)

    results = execute_query_blazegraph(config, query)["results"]["bindings"]
    for result in results:
        yield {'Source_table': result['Source_table']['value'],
               'Source_column': result['Source_column']['value'],
               'Source_column_type': result['Source_column_type']['value'],
               'Source_table_path': result['Source_table_path']['value'],
               'Target_table': result['Target_table']['value'],
               'Target_column': result['Target_column']['value'],
               'Certainty_score': result['Certainty_score']['value']}


def predict_entities(config, show_query):
    query = PREFIXES + """
    SELECT DISTINCT ?Entity ?Entity_data_type ?Table ?Table_path ?Column_1
    WHERE
    {
        <<?Column_1 lac:pkfk ?Column_2>>	lac:certainty	?Score	.
        ?Column_1	schema:name				?Entity					.
        ?Column_1	dct:isPartOf			?Table_id				.	
        ?Table_id	schema:name				?Table					;
                    lac:path                ?Table_path             .
        ?Column_1	schema:type				?Entity_data_type		.	
        ?Column_1	schema:totalVCount		?Total_values			.
        ?Column_1	schema:distinctVCount	?Distinct_values		.
        FILTER(?Total_values = ?Distinct_values)					.
    } ORDER BY ?Table
    """
    if show_query:
        print(query)

    results = execute_query_blazegraph(config, query)["results"]["bindings"]
    entities = {}
    for result in results:
        datatype = result['Entity_data_type']['value']
        if datatype == 'N':
            datatype = 'INT64'
        elif datatype == 'T':
            datatype = 'STRING'
        entities[result['Column_1']['value']] = {'name': result['Entity']['value'], 'datatype': datatype,
                                                 'Table': result['Table']['value'],
                                                 'Table_path': result['Table_path']['value']}

    return entities, execute_query(config, query).drop(['Column_1'], axis=1)
