from helpers.helper import execute_query, display_query

# --------------------------------------------KGFarm APIs---------------------------------------------------------------

def get_columns(config, table, dataset):
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
    SELECT DISTINCT ?Table ?Enrich_with ?Confidence_score ?Physical_joinable_table ?Table_path ?File_source ?Dataset ?Dataset_feature_view
    WHERE
    {
        # {
        <<?Primary_column_id        data:hasPrimaryKeyForeignKeySimilarity          ?Foreign_column_id>> data:withCertainty ?Confidence_score	.
        # }
        # UNION
        # {
        #  <<?Primary_column_id       data:hasDeepPrimaryKeyForeignKeySimilarity      ?Foreign_column_id>> data:withCertainty ?Confidence_score	. 
        #  FILTER(?Confidence_score >=1.0)
        #}
        
        ?Primary_column_id	        kglids:isPartOf			                    ?Primary_table_id	                        ;
                                    entity:name                                 ?Entity                                     ; 
                                    schema:name                                 ?join_key_name                              .    
        {
        ?Primary_table_id           kgfarm:hasDefaultEntity                     ?Primary_column_id                          .
        }
        UNION
        {
        ?Primary_table_id           kgfarm:hasMultipleEntities                  ?Primary_column_id                          .   
        }
        
        ?Primary_table_id           featureView:name                            ?Enrich_with                                 ;
                                    schema:name                                 ?Physical_joinable_table                    ;
                                    data:hasFilePath                            ?File_source                                ;
                                    kglids:isPartOf                             ?Dataset_feature_view_id                    .
        
        ?Dataset_feature_view_id    schema:name                                 ?Dataset_feature_view                       .
        
        
        ?Foreign_column_id          kglids:isPartOf                             ?Foreign_table_id                           ;
                                    schema:name                                 ?join_key_name                              .
        
        ?Foreign_table_id           schema:name                                 ?Table                                      ;
                                    data:hasFilePath                            ?Table_path                                 ;
                                    kglids:isPartOf                             ?Dataset_id                                 .
                                
        ?Dataset_id                 schema:name                                 ?Dataset                                    .                                        
                                
    } ORDER BY DESC(?Confidence_score)"""
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
SELECT DISTINCT (?Candidate_entity_name as ?Primary_column) ?Candidate_entity_dtype (?File_source as ?Primary_table) (?distinct_values/?total_values as ?Primary_key_uniqueness_ratio) (?Candidate_entity_id as ?Primary_column_id) (?Table_id as ?Primary_table_id) #?Total_number_of_columns
WHERE
    {
        ?Candidate_entity_id    rdf:type                    kglids:Column           ;   
                                schema:name                 ?Candidate_entity_name  ;
                                kglids:isPartOf             ?Table_id               ;
                                data:hasTotalValueCount     ?total_values           ;
                                data:hasDistinctValueCount  ?distinct_values        ;
                                data:hasMissingValueCount   ?missing_values         ;
                                data:hasDataType            ?Type                   .  
                                
        ?Table_id               schema:name                 ?File_source            ; 
                                kglids:isPartOf             ?All_columns            .
        
                
        
        FILTER(?missing_values = 0)                     # i.e. no missing values                               
        FILTER(?distinct_values/?total_values >= 0.95)  # high uniqueness         
        FILTER(?Type != 'T_date')                       # avoid timestamps            
    
        # convert dtype in feast format
        BIND(IF(REGEX(?Type, 'N'),'INT64','STRING') as ?Candidate_entity_dtype)  
        {
            SELECT ?Table_id (COUNT(?all_columns) as ?Total_number_of_columns)
            WHERE
            {
                ?Table_id       rdf:type        kglids:Table    .
                ?all_columns    kglids:isPartOf ?Table_id       .

            } GROUP BY ?Table_id
        }

        FILTER(?Total_number_of_columns > 2)
    } ORDER BY DESC(?Primary_table_id)"""
    return execute_query(config, query)


def get_number_of_relations(config, column_id: str):
    query = """
SELECT (COUNT(?All_columns) as ?Number_of_relations)
WHERE
{
    <%s> schema:name ?Column_name               .
    
    ?All_columns    schema:name ?Column_name    .
    
    ?All_columns    data:hasPrimaryKeyForeignKeySimilarity  ?Foreign_column .
}    

    """ % column_id
    return execute_query(config, query, return_type='json')


def get_pkfk_relations(config):
    query = """
    SELECT DISTINCT ?Primary_table_id ?Primary_column_id  (?Distinct_values/?Total_values as ?Primary_key_uniqueness_ratio) ?Primary_table ?Primary_column
    WHERE
    {
        <<?Primary_column_id    data:hasPrimaryKeyForeignKeySimilarity  ?Foreign_column_id>> data:withCertainty  ?Pkfk_score  .
    
        ?Primary_column_id      schema:name                 ?Primary_column     ;
                                kglids:isPartOf             ?Primary_table_id   ;
                                data:hasTotalValueCount     ?Total_values       ;
                                data:hasDistinctValueCount  ?Distinct_values    ;
                                data:hasMissingValueCount   ?Missing_values     .

        FILTER(?Missing_values = 0) 
        
        ?Foreign_column_id      schema:name                 ?Foreign_column     ;
                                kglids:isPartOf             ?Foreign_table_id   .
        
        ?Foreign_table_id       schema:name                 ?Foreign_table      .
        ?Primary_table_id       schema:name                 ?Primary_table      .
        {
            SELECT (?Table_id as ?Primary_table_id) (COUNT(?all_columns) as ?Total_number_of_columns)
            WHERE
            {
                ?Table_id       rdf:type        kglids:Table    .
                ?all_columns    kglids:isPartOf ?Table_id       .

            } GROUP BY ?Table_id
        }

        FILTER (?Total_number_of_columns > 2)
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
