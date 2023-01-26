from helpers.helper import execute_query, display_query


def get_columns(config, table, dataset, show_query):
    query = """
    SELECT DISTINCT ?Column	
    WHERE
    {
        ?Table_id			schema:name		"%s"				                    .
        ?Dataset_id			schema:name		"%s"				                    .
        ?Table_id			kglids:isPartOf	?Dataset_id								.
        ?Column_id			kglids:isPartOf	?Table_id								;
                            schema:name		?Column									.
    }""" % (table, dataset)
    if show_query:
        display_query(query)
    return execute_query(config, query)['Column'].tolist()


def predict_entities(config, show_query):
    query = """
        SELECT DISTINCT ?Entity ?Entity_data_type ?File_source ?File_source_path ?Dataset
    WHERE
    {
      <<?Entity_id 	data:hasPrimaryKeyForeignKeySimilarity  ?column_id_1>> data:withCertainty  ?score  .
      ?Entity_id	data:hasTotalValueCount                 ?Total_values			;
      				data:hasDistinctValueCount	            ?Distinct_values		;
                    schema:name				                ?Entity					;
      				data:hasDataType			            ?data_type		        .
      
      BIND(IF(REGEX(?data_type, 'N'),'INT64','STRING') as ?Entity_data_type) . 
      
      FILTER(?Total_values = ?Distinct_values)					    
      
      ?column_id_1	kglids:isPartOf			                ?Joinable_table_id		.       
      
      ?Entity_id	kglids:isPartOf			                ?Table_id				.
      ?Table_id     schema:name                             ?File_source            ;
                    data:hasFilePath				        ?File_source_path       ;
                    kglids:isPartOf                         ?Dataset_id             .
      
      ?Dataset_id   schema:name                             ?Dataset                .
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_all_tables(config, show_query):
    query = """
    SELECT DISTINCT (?Table_name as ?File_source) (?Table_path as ?File_source_path) (?Dataset_name as ?Dataset)
    WHERE
    {
        ?Table      rdf:type                kglids:Table    ;
                    schema:name             ?Table_name     ;
                    data:hasFilePath        ?Table_path     ;
                    kglids:isPartOf         ?Dataset_id     .
        
        ?Dataset_id schema:name             ?Dataset_name   .
    }"""
    if show_query:
        display_query(query)
    return execute_query(config, query)


def get_enrichable_tables(config, show_query):
    query = """
    SELECT DISTINCT ?Table ?Entity ?Path_to_table ?Dataset
    WHERE
    {
      <<?Entity_id 			data:hasPrimaryKeyForeignKeySimilarity  ?column_id_1>> data:withCertainty ?Score	.
      ?Entity_id			data:hasTotalValueCount		            ?Total_values			;
      						data:hasDistinctValueCount              ?Distinct_values		;
                    		schema:name				                ?Entity					;
      						data:hasDataType		                ?Entity_data_type		;
                    	    kglids:isPartOf			                ?Table_id				.
      
      FILTER(?Total_values = ?Distinct_values)					    		
      	
      ?Table_id     		schema:name                             ?File_source            ;
      						data:hasFilePath				        ?File_source_path       .
      
      ?column_id_1			kglids:isPartOf			                ?Joinable_table_id		.
      
      ?Joinable_table_id	schema:name				                ?Table      			;
                            data:hasFilePath				        ?Path_to_table			;
                            kglids:isPartOf                         ?Dataset_id             .
                
      ?Dataset_id           schema:name                             ?Dataset                .            
      FILTER(?Table_id != ?Joinable_table_id)					    		
    } ORDER BY DESC (?Entity) """
    if show_query:
        display_query(query)

    return execute_query(config, query)
#############################################################################


def get_INDs(config, show_query: bool = False):
    #SELECT (?Foreign_table_name as ?Foreign_table) (?Name_A as ?Foreign_key) ?A (?Primary_table_name as ?Primary_table) (?Name_B as ?Primary_key) ?B

    query = """
    SELECT(strbefore(?Foreign_table_name, '.csv') as ?Foreign_table) (?Name_A as ?Foreign_key) ?A(strbefore(?Primary_table_name, '.csv') as ?Primary_table) (?Name_B as ?Primary_key) ?B
    WHERE
    {
    ?B          schema:name                 ?Name_B                 ;
                kglids:isPartOf             ?Table_B                .
    <<?B        data:hasInclusionDependency ?A >>    data:withCertainty  ?Score  .
        
    ?A          kglids:isPartOf             ?Table_A                ;
                schema:name                 ?Name_A                 .
        
    ?Table_B    schema:name                 ?Primary_table_name     .
    ?Table_A    schema:name                 ?Foreign_table_name     .
    FILTER(?Score>0.60)
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_content_similar_pairs(config, show_query: bool = False):
    query = """
    SELECT(strbefore(?Foreign_table_name, '.csv') as ?Foreign_table) (?Name_A as ?Foreign_key) ?A(strbefore(?Primary_table_name, '.csv') as ?Primary_table) (?Name_B as ?Primary_key) ?B
    WHERE
    {
    ?B          schema:name                 ?Name_B                                         ;
                kglids:isPartOf             ?Table_B                                        .
    <<?B        data:hasContentSimilarity ?A >>    data:withCertainty  ?Score  .

    ?A          kglids:isPartOf             ?Table_A                                        ;
                schema:name                 ?Name_A                                         .

    ?Table_B    schema:name                 ?Primary_table_name                             .
    ?Table_A    schema:name                 ?Foreign_table_name                             .
    FILTER(?Score<=(1-0.65))
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)

def get_distinct_dependent_values(config, show_query: bool = False):
    query = """
    SELECT ?A ?B (?Distinct_values/?Total_values AS ?F1)
    WHERE
    {   
        ?A  data:hasTotalValueCount     ?Total_values       ;
            data:hasDistinctValueCount  ?Distinct_values    .  
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_content_similarity(config, show_query: bool = False):
    query = """
    SELECT ?A ?B ((1-?Score) AS ?F2)
    WHERE
    {   
        <<?B data:hasContentSimilarity           ?A>>    data:withCertainty  ?Score  .                         
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_column_name_similarity(config, show_query: bool = False):
    query = """
    SELECT ?A ?B ?F6
    WHERE
    {                                    
        <<?B data:hasLabelSimilarity  ?A>>    data:withCertainty  ?Score  .
        BIND(IF((?Score=1),1,0) as ?F6) .
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_range(config, show_query: bool = False):
    query = """
    SELECT ?A ?B ?F8
    WHERE
    {
    ?B  schema:name                 ?Name_B .
    ?A  schema:name                 ?Name_A .
    ?A  kglids:isPartOf             ?tA.
    ?B  kglids:isPartOf             ?tB.
    ?A  data:hasMaxValue            ?maxA .
    ?A  data:hasMinValue            ?minA .
    ?B  data:hasMaxValue            ?maxB .
    ?B  data:hasMinValue            ?minB .
    
    BIND(IF((?maxB=?maxA && ?minB=?minA),1,0) as ?F8) .
    FILTER(?tA != ?tB) .   
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_range_with_value(config, A,B,show_query: bool = False):
    query = """
    SELECT ?F8
    WHERE
    {
    ?B  schema:name                 ?Name_B .
    ?A  schema:name                 ?Name_A .
    ?A  kglids:isPartOf             ?tA.
    ?B  kglids:isPartOf             ?tB.
    ?A  data:hasMaxValue            ?maxA .
    ?A  data:hasMinValue            ?minA .
    ?B  data:hasMaxValue            ?maxB .
    ?B  data:hasMinValue            ?minB .

    BIND(IF((?maxB=?maxA && ?minB=?minA),1,0) as ?F8) .
    FILTER(?tA != ?tB  && ?A='"""+A+"""'  && ?B='"""+B+"""') .   
    }"""
    if show_query:
        display_query(query)
    x=execute_query(config, query)
    #print(query)
    return execute_query(config, query)


def get_typical_name_suffix(config, show_query: bool = False):
    query = """
    SELECT ?A ?B ?F9
    WHERE
    {
    ?B  schema:name                 ?Name_B .
    ?A  schema:name                 ?Name_A .
    
    BIND(IF(REGEX(?Name_A, 'id$', "i" )||REGEX(?Name_A, 'key$', "i" )||REGEX(?Name_A, 'num$', "i" ),1,0) as ?F9) .   
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_table_size_ratio(config, show_query: bool = False):
    query = """
    SELECT ?A ?B  (?Rows_A/?Rows_B AS ?F10)
    WHERE
    {
    ?B  data:hasTotalValueCount     ?Rows_A .
    ?A  data:hasTotalValueCount     ?Rows_B .
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


