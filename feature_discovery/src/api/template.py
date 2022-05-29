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
#############################################################################


def get_INDs(config, show_query: bool = False):
    query = """
    SELECT ?A ?B 
    WHERE
    {
    ?B  data:hasInclusionDependency ?A      .  
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_distinct_dependent_values(config, show_query: bool = False):
    query = """
    SELECT ?A ?B (?Distinct_values/?Total_values AS ?F1)
    WHERE
    {   
        ?B  data:hasInclusionDependency ?A                  .
        
        ?A  data:hasTotalValueCount     ?Total_values       ;
            data:hasDistinctValueCount  ?Distinct_values    .  
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_content_similarity(config, show_query: bool = False):
    query = """
    SELECT ?A ?B (?Score AS ?F2)
    WHERE
    {   
        ?B  data:hasInclusionDependency ?A                                  .
        <<?B data:hasContentSimilarity  ?A>>    data:withCertainty  ?Score  .
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_column_name_similarity(config, show_query: bool = False):
    query = """
    SELECT ?A ?B (?Score AS ?F6)
    WHERE
    {   
        ?B  data:hasInclusionDependency ?A                                  .
        <<?B data:hasSemanticSimilarity  ?A>>    data:withCertainty  ?Score  .
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_range(config, show_query: bool = False):
    query = """
    PREFIX kglids: <http://kglids.org/ontology/>
    SELECT ?A ?B ?F8
    WHERE
    {
    ?B  data:hasInclusionDependency ?A      ;
        schema:name                 ?Name_A .
    ?A  schema:name                 ?Name_B .
    ?A  kglids:isPartOf             ?tA.
    ?B  kglids:isPartOf             ?tB.
    ?A  data:hasMaxValue            ?maxA .
    ?A  data:hasMinValue            ?minA .
    ?B  data:hasMaxValue            ?maxB .
    ?B  data:hasMinValue            ?minB .
    
    BIND(IF((?maxB>=?maxA && ?minB<=?minA),1,0) as ?F8) .
    FILTER(?tA != ?tB) .   
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_typical_name_suffix(config, show_query: bool = False):
    query = """
    SELECT ?A ?B ?F9
    WHERE
    {
    ?B  data:hasInclusionDependency ?A      ;
        schema:name                 ?Name_A .
    ?A  schema:name                 ?Name_B .
    
    BIND(IF(REGEX(?Name_A, 'id$', "i" )||REGEX(?Name_A, 'key$', "i" )||REGEX(?Name_A, 'num_$', "i" ),1,0) as ?F9) .   
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_table_size_ratio(config, show_query: bool = False):
    query = """
    SELECT ?A ?B  (?Rows_A/?Rows_B AS ?F10)
    WHERE
    {
    ?B  data:hasInclusionDependency ?A      ;
        data:hasTotalValueCount     ?Rows_A .
    ?A  data:hasTotalValueCount     ?Rows_B .
    } 
    """
    if show_query:
        display_query(query)

    return execute_query(config, query)


