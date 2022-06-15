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


def get_default_entities(config, show_query):
    query = """
    SELECT ?Entity_name ?Entity_data_type ?Physical_column (?Table as ?Physical_table) (?Score as ?Uniqueness_ratio)
    {
    <<?Table_id kgfarm:hasDefaultEntity ?Entity_id>>    kgfarm:confidence   ?Score  .
    ?Entity_id  entity:name         ?Entity_name                                    ;
                schema:name         ?Physical_column                                ;
                data:hasDataType    ?Physical_column_data_type                      .
    ?Table_id   schema:name         ?Table                                          ;
                featureView:name    ?Feature_view                                   .
    
    BIND(IF(REGEX(?Physical_column_data_type, 'N'),'INT64','STRING') as ?Entity_data_type)
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_multiple_entities(config, show_query):
    query = """
    SELECT ?Entity_name ?Entity_data_type ?Physical_column (?Table as ?Physical_table) (?Score as ?Uniqueness_ratio)
    {
    <<?Table_id kgfarm:hasMultipleEntities ?Entity_id>>    kgfarm:confidence   ?Score  .
    ?Entity_id  entity:name         ?Entity_name                                    ;
                schema:name         ?Physical_column                                ;
                data:hasDataType    ?Physical_column_data_type                      .
    ?Table_id   schema:name         ?Table                                          ;
                featureView:name    ?Feature_view                                   .

    BIND(IF(REGEX(?Physical_column_data_type, 'N'),'INT64','STRING') as ?Entity_data_type)
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_feature_views(config, show_query):
    query = """
    SELECT ?Feature_view (?Entity_name as ?Entity) ?Entity_data_type ?Physical_column ?Physical_column_data_type  ?Physical_table ?File_source (?Score as ?Uniqueness_ratio)
    {
        {       
            <<?Table_id kgfarm:hasDefaultEntity ?Entity_id>>    kgfarm:confidence   ?Score  .
        }
        UNION
        {
            <<?Table_id kgfarm:hasMultipleEntities ?Entity_id>>    kgfarm:confidence   ?Score  .
        }
        
        ?Entity_id      entity:name         ?Entity_name                        ;
                        data:hasDataType    ?Physical_column_data_type          ;  
                        schema:name         ?Physical_column                    .
        
        ?Table_id       featureView:name    ?Feature_view                       ;
                        data:hasFilePath    ?File_source                        ;
                        schema:name         ?Physical_table                     .
        
        BIND(IF(REGEX(?Physical_column_data_type, 'N'),'INT64','STRING') as ?Entity_data_type)
    } ORDER BY ?Feature_view"""
    if show_query:
        display_query(query)
    return execute_query(config, query)


def get_feature_views_without_entities(config, show_query):
    query = """
    SELECT ?Feature_view ?Physical_table ?File_source 
    WHERE
    {
        ?Table_id   kgfarm:hasNoEntity  ?Uniqueness_ratio           ;
                    featureView:name    ?Feature_view               ;
                    data:hasFilePath    ?File_source                ;
                    schema:name         ?Physical_table             .
    }"""
    if show_query:
        display_query(query)
    return execute_query(config, query)


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

# --------------------------------------------Farm Builder-------------------------------------------------------------

def get_table_ids(config):
    query = """
    SELECT DISTINCT ?Table_id
    WHERE
    {
      ?Table_id rdf:type  kglids:Table  .
    } """
    return execute_query(config, query)


def detect_entities(config):
    query = """ 
    SELECT DISTINCT (?Candidate_entity_name as ?Primary_column) ?Candidate_entity_dtype (?File_source as ?Primary_table) (?distinct_values/?total_values as ?Primary_key_uniqueness_ratio) (?Candidate_entity_id as ?Primary_column_id) (?Table_id as ?Primary_table_id)
    WHERE
    {
        ?Candidate_entity_id    rdf:type                    kglids:Column           ;   
                                schema:name                 ?Candidate_entity_name  ;
                                kglids:isPartOf             ?Table_id               ;
                                data:hasTotalValueCount     ?total_values           ;
                                data:hasDistinctValueCount  ?distinct_values        ;
                                data:hasMissingValueCount   ?missing_values         ;
                                data:hasDataType            ?Type                   .  
                                
        ?Table_id               schema:name                 ?File_source            .  
    
        FILTER(?missing_values = 0)                     # i.e. no missing values                               
        FILTER(?distinct_values/?total_values >= 0.95)  # high uniqueness         
        FILTER(?Type != 'T_date')                       # avoid timestamps            
    
        # convert dtype in feast format
        BIND(IF(REGEX(?Type, 'N'),'INT64','STRING') as ?Candidate_entity_dtype)  
    } ORDER BY DESC(?Primary_table_id)"""
    return execute_query(config, query)


def get_number_of_relations(config, column_id: str):
    query = """
    SELECT (COUNT(?relation) as ?Number_of_relations)
    WHERE
    {
        <<<%s> ?relation ?column_id>> data:withCertainty ?Score.
    }
    """ % column_id
    return execute_query(config, query, return_type='json')


def get_pkfk_relations(config):
    query = """
    # SELECT DISTINCT ?Primary_table ?Primary_column ?Foreign_table ?Foreign_column ?Pkfk_score (?Distinct_values/?Total_values as ?Primary_key_uniqueness_ratio) ?Primary_table_id ?Primary_column_id ?Foreign_table_id ?Foreign_column_id
    SELECT DISTINCT ?Primary_table_id ?Primary_column_id  (?Distinct_values/?Total_values as ?Primary_key_uniqueness_ratio) ?Primary_table ?Primary_column
    WHERE
    {
        <<?Primary_column_id    data:hasPrimaryKeyForeignKeySimilarity  ?Foreign_column_id>> data:withCertainty  ?Pkfk_score  .
    
        ?Primary_column_id      schema:name                 ?Primary_column     ;
                                kglids:isPartOf             ?Primary_table_id   ;
                                data:hasTotalValueCount     ?Total_values       ;
                                data:hasDistinctValueCount  ?Distinct_values    . 
        
        ?Foreign_column_id      schema:name                 ?Foreign_column     ;
                                kglids:isPartOf             ?Foreign_table_id   .
        
        ?Foreign_table_id       schema:name                 ?Foreign_table      .
        ?Primary_table_id       schema:name                 ?Primary_table      .
    }"""
    return execute_query(config, query)

# --------------------------------------------FKC Extractor-------------------------------------------------------------


def get_INDs(config, show_query: bool = False):
    query = """
    SELECT (strbefore(?Foreign_table_name, '.csv') as ?Foreign_table) (?Name_A as ?Foreign_key) ?A (strbefore(?Primary_table_name, '.csv') as ?Primary_table) (?Name_B as ?Primary_key) ?B 
    WHERE
    {
    ?B          data:hasInclusionDependency ?A                      ;
                schema:name                 ?Name_B                 ;
                kglids:isPartOf             ?Table_B                .
        
    ?A          kglids:isPartOf             ?Table_A                ;
                schema:name                 ?Name_A                 .
        
    ?Table_B    schema:name                 ?Primary_table_name     .
    ?Table_A    schema:name                 ?Foreign_table_name     .
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
    }"""
    if show_query:
        display_query(query)

    return execute_query(config, query)
