from helpers.helper import execute_query_blazegraph, execute_query

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
    SELECT DISTINCT ?Entity_id ?Entity ?Entity_data_type ?FileSource ?FileSource_path 
    WHERE
    {
      <<?Entity_id  lac:pkfk ?Column_2>>	lac:certainty	?Score	.
      ?Entity_id	schema:name				?Entity					.
      ?Column_2	    dct:isPartOf			?Table_id				.	
      ?Table_id	    schema:name				?FileSource				;
                    lac:path                ?FileSource_path        .
      ?Entity_id	schema:type				?Entity_data_type		.	
      ?Entity_id	schema:totalVCount		?Total_values			.
      ?Entity_id	schema:distinctVCount	?Distinct_values		.
      FILTER(?Total_values = ?Distinct_values)					    .
    }
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
        if result['Entity_id']['value'] in entities:
            file_source = entities.get(result['Entity_id']['value'])['FileSource_path']
            file_source.append(result['FileSource_path']['value'])
            entities[result['Entity_id']['value']] = {'name': result['Entity']['value'], 'datatype': datatype,
                                                      'FileSource_path': file_source}
        else:
            entities[result['Entity_id']['value']] = {'name': result['Entity']['value'], 'datatype': datatype,
                                                      'FileSource_path': [result['FileSource_path']['value']]}

    return entities, execute_query(config, query).drop(['Entity_id'], axis=1). \
        replace(['N', 'T'], ['INT64', 'STRING'])


def predict_features(config, table, dataset, show_query):
    print('table: ', table)
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
    } ORDER BY ?Joinable_table """ % (table, dataset)
    if show_query:
        print(query)
    df = execute_query(config, query)
    join_key_id = df['Join_key_id'].iloc[0]
    joinable_table = df['Joinable_table'].iloc[0]
    print('Joinable table: ', joinable_table)
    get_columns(config, joinable_table, dataset)
    x = [i for i in get_columns(config, joinable_table, dataset) if i not in get_columns(config, table, dataset)]
    # a = list(set(df['Column_x'].tolist()))
    # b = list(set(df['Column_y'].tolist()))
    # columns_to_join = [i for i in list(set(df['Column_y'].tolist())) if i not in list(set(df['Column_x'].tolist()))]
    print('difference: ', x)
