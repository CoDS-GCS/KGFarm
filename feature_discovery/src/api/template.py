from helpers.helper import execute_query, display_query


def get_columns(config, table, dataset, show_query):
    query = """
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
    if show_query:
        display_query(query)
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
      
      ?column_id_1	dct:isPartOf			?Joinable_table_id		.
      #             schema:name             ?Foreign_key            .
      
      # FILTER(?Entity = ?Foreign_key)       
      
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


def get_all_tables(config, show_query):
    query = """
    SELECT DISTINCT (?Table_name as ?File_source) (?Table_path as ?File_source_path) (?Dataset_name as ?Dataset)
    WHERE
    {
        ?Table      rdf:type        lac:table       ;
                    schema:name     ?Table_name     ;
                    lac:path        ?Table_path     ;
                    dct:isPartOf    ?Dataset_id     .
        
        ?Dataset_id schema:name     ?Dataset_name   .
    }
    """
    if show_query:
        display_query(query)
    return execute_query(config, query)


def get_enrichable_tables(config, show_query):
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
      #                      schema:name             ?Foreign_key           .
      
      # FILTER(?Entity = ?Foreign_key)                                      .
      
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
