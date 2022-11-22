from helpers.helper import execute_query, display_query


# --------------------------------------------KGFarm APIs (SELECT queries)----------------------------------------------


def get_table_path(config, table, dataset):
    query = """
    SELECT ?Table_path
    WHERE
    {
    ?Table_id   schema:name         "%s";
                kglids:isPartOf     ?Dataset_id     ;
                data:hasFilePath    ?Table_path     .
    ?Dataset_id schema:name         "%s".
    }""" % (table, dataset)
    return str(execute_query(config, query)['Table_path'][0])


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
        # format in feast requires fv: feature_name, hence features with ':' need to be renamed or removed. 
        FILTER(!regex((?Column), (':')))
    }""" % (table, dataset)
    return execute_query(config, query)['Column'].tolist()


def get_entities(config, show_query):
    query = """
    SELECT ?Entity_name (?Physical_column_data_type as ?Entity_data_type) ?Physical_column (?Table as ?Physical_table) (?Score as ?Uniqueness_ratio)
    {
        {
            <<?Table_id kgfarm:hasDefaultEntity ?Entity_id>>        kgfarm:confidence   ?Score  .
        }
        UNION
        {
            <<?Table_id kgfarm:hasMultipleEntities ?Entity_id>>     kgfarm:confidence   ?Score  .
        }
    
        ?Entity_id  entity:name         ?Entity_name                                            ;
                    schema:name         ?Physical_column                                        ;
                    data:hasDataType    ?Physical_column_data_type                              .
        ?Table_id   schema:name         ?Table                                                  .
    }Order by DESC(?Uniqueness_ratio)"""

    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_feature_views_with_one_or_no_entity(config, show_query):
    query = """
    # Select feature views with 1 or no entity
    SELECT ?Feature_view ?Entity ?Physical_column ?Physical_table ?File_source
    WHERE
    {
        ?Table_id       rdf:type                    kglids:Table        ;
                        featureView:name            ?Feature_view       ;
                        schema:name                 ?Physical_table     ;   
                        data:hasFilePath            ?File_source        .
        OPTIONAL
        {
            ?Table_id   kgfarm:hasDefaultEntity     ?Column_id          .
            ?Column_id  entity:name                 ?Entity             ;
                        schema:name                 ?Physical_column    .
        }
        MINUS # remove feature view with multiple entities
        {
            ?Table_id   kgfarm:hasMultipleEntities  ?Column_id          .
            ?Column_id  entity:name                 ?Entity             ;
                        schema:name                 ?Physical_column    .
        }
    }"""
    if show_query:
        display_query(query)
    return execute_query(config, query)


def get_feature_views_with_multiple_entities(config, show_query):
    query = """
    # Select feature views with multiple entities
    SELECT ?Feature_view ?Entity ?Physical_column ?Physical_table ?File_source
    WHERE
    {
        ?Table_id       rdf:type                    kglids:Table        ;
                        featureView:name            ?Feature_view       ;
                        schema:name                 ?Physical_table     ;   
                        data:hasFilePath            ?File_source        ;
                        kgfarm:hasMultipleEntities  ?Column_id          .
        
        ?Column_id      entity:name                 ?Entity             ;
                        schema:name                 ?Physical_column    .
    } ORDER BY ?Feature_view"""
    if show_query:
        display_query(query)
    return execute_query(config, query)


def search_enrichment_options(config, show_query):
    query = """
    SELECT DISTINCT ?Table ?Enrich_with (?Confidence_score as ?Joinability_strength) (?join_key_name as ?Join_key) ?Physical_joinable_table ?Table_path ?File_source ?Dataset ?Dataset_feature_view
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
        
        ?Primary_table_id           featureView:name                            ?Enrich_with                                ;
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
                                
    } ORDER BY DESC(?Confidence_score) ?Table ?Feature_view"""
    if show_query:
        display_query(query)

    return execute_query(config, query)


def get_optional_entities(config, show_query):
    query = """
    SELECT ?Feature_view ?Entity ?Current_physical_representation ?Optional_physical_representation ?Data_type (?Score as ?Uniqueness_ratio) ?Physical_table
    WHERE
    {
        <<?Table_id         kgfarm:hasEntity        ?Column_id>> kgfarm:confidence ?Score   .
        
        ?Column_id          schema:name             ?Optional_physical_representation       ;
                            data:hasDataType        ?Data_type                              .
        
        ?Table_id           kgfarm:hasDefaultEntity ?Default_column_id                      ;                   
                            featureView:name        ?Feature_view                           ;
                            schema:name             ?Physical_table                         .
        
        ?Default_column_id  entity:name             ?Entity                                 ;
                            schema:name             ?Current_physical_representation        .
     
        FILTER(?Column_id != ?Default_column_id) 
    } ORDER BY ?Feature_view DESC(?Uniqueness_ratio) ?Optional_physical_representation"""
    if show_query:
        display_query(query)
    return execute_query(config, query)


# TODO: add filter for table and dataset
def recommend_feature_transformations(config, show_query):
    query = """
    SELECT DISTINCT ?Transformation ?Package ?Function ?Library ?Feature ?Feature_view ?Table ?Dataset ?Pipeline ?Author ?Written_on ?Pipeline_url  
    WHERE
    {
    # query pipeline-default graph
    ?Pipeline_id            rdf:type                kglids:Pipeline     ;
                            kglids:isPartOf         ?Dataset_id         ;
                            rdfs:label              ?Pipeline           ;
                            pipeline:isWrittenBy    ?Author             ;
                            pipeline:isWrittenOn    ?Written_on         ;
                            pipeline:hasSourceURL   ?Pipeline_url       . 
    
    # querying named-graphs for pipeline               
    GRAPH ?Pipeline_id
    {
        ?Statement          pipeline:callsClass     ?Class_id           .
        ?Statement_2        pipeline:callsFunction  ?Function_id        ;
                            pipeline:readsColumn    ?Column_id          .
        # <<?Statement_2    pipeline:hasParameter   ?o>> ?p1 ?o1        .
    }
    
    # querying the link between default-named graphs relationships
    ?Class_id               kglids:isPartOf          <http://kglids.org/resource/library/sklearn/preprocessing> .
    ?Function_id            kglids:isPartOf          ?Class_id          . 
    <http://kglids.org/resource/library/sklearn/preprocessing> kglids:isPartOf ?Library_id                      .
    
    # query data-items
    ?Dataset_id             schema:name             ?Dataset            .
    ?Column_id              schema:name             ?Feature            ;
                            kglids:isPartOf         ?Table_id           .
    ?Table_id               schema:name             ?Table              .
    
    # querying the link between Farm and LiDS graph
    ?Table_id               featureView:name         ?Feature_view      .
    
    # beautify output
    BIND(STRAFTER(str(?Library_id), str(lib:)) as ?Library)             
    BIND(STRAFTER(str(?Class_id), str('http://kglids.org/resource/library/sklearn/preprocessing/')) as ?Transformation)     
    BIND(STRAFTER(str(?Class_id), str('http://kglids.org/resource/library/sklearn/')) as ?P)  
    BIND(STRBEFORE(str(?P), str('/')) as ?Package)                       
    BIND(STRAFTER(str(?Function_id), str('http://kglids.org/resource/library/sklearn/preprocessing/')) as ?F)               
    BIND(REPLACE(?F, '/', '.', 'i') AS ?F1)                               
    BIND(CONCAT(STR( ?F1 ), '( )') AS ?Function)                                     
    
    # sort by dataset names and transformations
    } ORDER BY ?Dataset ?Table ?Transformation ?Author"""

    if show_query:
        display_query(query)
    return execute_query(config, query)


def search_entity_table(config, columns):
    def generate_subquery():
        subquery = ''
        col_count = 0
        for column in columns:
            subquery = subquery + "\t?Column_id_{} schema:name '{}';\n\t\t\tkglids:isPartOf ?Table_id.\n\n". \
                format(col_count, column)
            col_count = col_count + 1
        return subquery

    query = """
    SELECT ?Table ?Table_id
    WHERE
    {
    %s
        ?Table_id   schema:name         ?Table.
    }
    """ % generate_subquery()
    return execute_query(config, query)


def get_physical_table(config, feature_view):
    query = """
    SELECT ?Table_id
    WHERE
    {
        ?Table_id featureView:name ?Feature_view    .
        FILTER(?Feature_view = '%s')
    }""" % feature_view
    return execute_query(config, query)['Table_id'][0]


def get_table_name(config, table_id):
    query = """
    SELECT ?Table
    WHERE
    {
        <%s>    schema:name ?Table .
    }
    """ % table_id
    return execute_query(config, query)['Table'][0]


def is_entity_column(config, feature, dependent_variable):
    query = """
    ASK
    {  
      ?y          kglids:isPartOf ?Table_id     ;
                  schema:name     "%s"          .
    
      ?Column_id  schema:name     "%s"          ;
                  entity:name     ?Entity       ;
                  kglids:isPartOf ?Table_id     . 
    }""" % (dependent_variable, feature)

    return execute_query(config, query, return_type='ask')


def identify_features(config, entity_name, target_name, show_query):
    query = """
    SELECT DISTINCT ?Entity ?Feature ?Physical_representation ?Feature_view ?Physical_table ?Number_of_rows ?File_source ?Dataset
    WHERE
    {
        ?Column_id  entity:name             ?Entity                     ;  
                    schema:name             ?Physical_representation    ;  
                    kglids:isPartOf         ?Table_id                   ;
                    data:hasTotalValueCount ?Number_of_rows             .               
            
        ?Table_id   schema:name             ?Physical_table             ;
                    featureView:name        ?Feature_view               ;
                    data:hasFilePath        ?File_source                ;
                    kglids:isPartOf         ?Dataset_id                 .
        
        ?Feature_id kglids:isPartOf         ?Table_id                   ;
                    schema:name             ?Feature                    .
                    
        ?Dataset_id schema:name             ?Dataset                    .
        
        # search entity
        FILTER regex(str(?Entity), "%s", "i")    # 'i' ignore case sensitivity
        FILTER regex(str(?Feature), "%s", "i")   # 'i' ignore case sensitivity
    } ORDER BY DESC(?Number_of_rows)""" % (entity_name, target_name)
    if show_query:
        display_query(query)
    return execute_query(config, query)


def get_features_to_drop(config, table_id, show_query):
    query = """
    SELECT DISTINCT ?Feature_to_drop
    WHERE
    {
        # query pipeline-default graph
        ?Pipeline_id        rdf:type                kglids:Pipeline     ;
                            kglids:isPartOf         ?Dataset_id         ;
                            rdfs:label              ?Pipeline           ;
                            pipeline:isWrittenBy    ?Author             ;
                            pipeline:isWrittenOn    ?Written_on         ;
                            pipeline:hasSourceURL         ?p            . 
    
        # querying named-graphs for pipeline               
        GRAPH ?Pipeline_id
        {
            ?Statement      pipeline:callsClass     ?Class_id           .
            ?Statement_2    pipeline:callsFunction  ?Function_id        ;
                            pipeline:readsColumn    ?Column_id          .
        }
    
        ?Column_id          schema:name             ?Feature_to_drop    ;
                            kglids:isPartOf         <%s>                .
        
        <%s>                kglids:isPartOf         ?Dataset_id         ;
                            schema:name             ?Table              .
    
        BIND(STRAFTER(str(?Statement_2), str(?Dataset_id)) as ?Call1)   .
        BIND(STRAFTER(str(?Call1), str('dataResource/')) as ?Call2)     .
        BIND(STRAFTER(str(?Call2), str('/s')) as ?Call)                 .
    
    
        BIND(STRAFTER(str(?Function_id), str(lib:)) as ?Function1)      .
        BIND(REPLACE(?Function1, '/', '.', 'i') AS ?Function)           .
    
        FILTER(?Function = 'pandas.DataFrame.drop') 
            
    } ORDER BY ?Pipeline xsd:integer(?Call) ?Table""" % (table_id, table_id)
    if show_query:
        display_query(query)
    return execute_query(config, query)


def get_data_cleaning_info(config, table_id, show_query):
    query = """
    SELECT DISTINCT ?Table ?Function ?Parameter ?Value ?Feature_view
    WHERE
    {
    # query pipeline-default graph
    ?Pipeline_id            rdf:type                kglids:Pipeline     ;
                            kglids:isPartOf         ?Dataset_id         ;
                            rdfs:label              ?Pipeline           ;
                            pipeline:isWrittenBy    ?Author             ;
                            pipeline:isWrittenOn    ?Written_on         ;
                            pipeline:hasSourceURL   ?Pipeline_url       .
    
    # querying named-graphs for pipeline               
    GRAPH ?Pipeline_id
    {
        ?Statement          pipeline:callsFunction  ?Function_id        .
        <<?Statement        pipeline:hasParameter   ?Parameter>> pipeline:withParameterValue ?Value        .
        ?Statement_2        pipeline:readsTable     <%s>                .          
    }
    
    <%s>                    schema:name             ?Table              ;
                            featureView:name        ?Feature_view       .

    # Methods for dealing with missing data
    FILTER(?Function = "pandas.DataFrame.interpolate" || ?Function = "pandas.DataFrame.fillna" || ?Function = "pandas.DataFrame.dropna")
    # Filter out None/False as a value and axis or inplace as a parameter
    FILTER(?Value != 'None' && ?Value != 'False' && ?Parameter != 'axis' && ?Parameter != 'inplace' && ?Value != 'DataFrame' && ?Value != '()' && ?Value != '[]' && ?Value != '{}' && ?Value != '')
    
    # beautify output
    BIND(STRAFTER(str(?Function_id), str(lib:)) as ?Function1)             
    BIND(REPLACE(?Function1, '/', '.', 'i') AS ?Function)                               
    } ORDER BY ?Function ?Pipeline""" % (table_id, table_id)
    if show_query:
        display_query(query)
    return execute_query(config, query)


def get_data_cleaning_recommendation(config, table_id, show_query=False, timeout=10000):  # data cleaning for unseen data
    query = """
    SELECT DISTINCT  ?Function ?Parameter ?Value ?Column_id ?Pipeline
    WHERE
    {
    # query pipeline-default graph
    ?Pipeline_id            rdf:type                kglids:Pipeline     ;
                            rdfs:label              ?Pipeline           .
    
    # querying named-graphs for pipeline               
    GRAPH ?Pipeline_id
    {
        ?Statement          pipeline:callsFunction  ?Function_id        .
        <<?Statement        pipeline:hasParameter   ?Parameter>> pipeline:withParameterValue ?Value        .
        ?Statement_2        pipeline:readsTable     <%s>                .   
        OPTIONAL
        {
            ?Statement      pipeline:readsColumn    ?Column_id          .
        }   
    }
    
    ?Function_id kglids:isPartOf <http://kglids.org/resource/library/pandas/DataFrame>                     .
    FILTER(?Value != 'None' && ?Value != 'False' && ?Parameter != 'axis' && ?Parameter != 'inplace' && ?Value != 'DataFrame' && ?Value != '()' && ?Value != '[]'  && ?Value != '' && ?Value !='NaN')
    # Methods for dealing with missing data
    FILTER(?Function = "pandas.DataFrame.interpolate" || ?Function = "pandas.DataFrame.fillna" || ?Function = "pandas.DataFrame.dropna")
    
    # beautify output
    BIND(STRAFTER(str(?Function_id), str(lib:)) as ?Function1)             
    BIND(REPLACE(?Function1, '/', '.', 'i') AS ?Function)                               
    } ORDER BY DESC(?Function) ?Pipeline """ % table_id
    if show_query:
        display_query(query)

    return execute_query(config, query, timeout=timeout)


# --------------------------------------KGFarm APIs (updation queries) via Governor-------------------------------------


def exists(config, feature_view):
    query = """ASK { ?Table_id featureView:name  '%s'}""" % feature_view
    return execute_query(config, query, return_type='ask')


def drop_feature_view(config, feature_view):
    query = """
    DELETE 
    {     
        ?Table_id featureView:name ?Feature_view            .
    }
    WHERE  { ?Table_id featureView:name ?Feature_view
             FILTER (?Feature_view = '%s') 
    }""" % feature_view
    execute_query(config, query, return_type='update')


def remove_current_physical_representation_of_an_entity(config, feature_view):
    query = """
    DELETE 
    {
        <<?Table_id   kgfarm:hasDefaultEntity     ?Column_id>> kgfarm:confidence ?Score     .
        ?Column_id    entity:name                 ?Entity                                   .
    }
    WHERE  
    {
        <<?Table_id     kgfarm:hasDefaultEntity     ?Column_id>> kgfarm:confidence ?Score   .
        ?Column_id      entity:name                 ?Entity                                 .
        ?Table_id       featureView:name            '%s'                                    .
    }""" % feature_view
    execute_query(config, query, return_type='update')


def insert_current_physical_representation_of_an_entity(config, feature_view, column, entity):
    query = """
    INSERT
    {
        <<?Table_id   kgfarm:hasDefaultEntity     ?Column_id>> kgfarm:confidence ?Score     .
        ?Column_id    entity:name                 '%s'                                      .
    }
    WHERE  
    {
        <<?Table_id     kgfarm:hasEntity     ?Column_id>> kgfarm:confidence ?Score          .
        ?Column_id      schema:name                 '%s'                                    .
        ?Table_id       featureView:name            '%s'                                    .
    }""" % (entity, column, feature_view)
    execute_query(config, query, return_type='update')


# --------------------------------------------Farm Builder--------------------------------------------------------------


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
        # ignore event timestamps
        FILTER(?Candidate_entity_name != 'event_timestamp')
    } ORDER BY DESC(?Primary_table_id)"""
    return execute_query(config, query)


def get_number_of_relations(config, column_id: str):
    query = """
    SELECT (COUNT(?All_columns) as ?Number_of_relations)
    WHERE
    {
        <%s> schema:name ?Column_name                                           .
        
        ?All_columns    schema:name ?Column_name                                .
        
        ?All_columns    data:hasPrimaryKeyForeignKeySimilarity  ?Foreign_column .
    }""" % column_id
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
        # ignore event timestamps
        FILTER(?Primary_column != 'event_timestamp')
        FILTER(?Foreign_column != 'event_timestamp')
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
        ?B  data:hasInclusionDependency ?A                                      .
        <<?B data:hasSemanticSimilarity  ?A>>    data:withCertainty  ?Score     .
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
    ?A  kglids:isPartOf             ?tA     .
    ?B  kglids:isPartOf             ?tB     .
    ?A  data:hasMaxValue            ?maxA   .
    ?A  data:hasMinValue            ?minA   .
    ?B  data:hasMaxValue            ?maxB   .
    ?B  data:hasMinValue            ?minB   .
    
    BIND(IF((?maxB>=?maxA && ?minB<=?minA),1,0) as ?F8) 
    FILTER(?tA != ?tB)  
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


# --------------------------------------------Transformation recommender------------------------------------------------


def get_transformations_on_columns(config):
    query = """PREFIX preprocessing:   <http://kglids.org/resource/library/sklearn/preprocessing/>
    SELECT DISTINCT  ?Transformation ?Column_id
    WHERE
    {
        # query pipeline-default graph
        ?Pipeline_id        rdf:type                kglids:Pipeline     ;
                            rdfs:label              ?Pipeline           .
        # querying named-graphs for pipeline               
        GRAPH ?Pipeline_id
        {
            ?Statement      pipeline:callsFunction  ?Function_id        ;
                            pipeline:readsColumn    ?Column_id          .
            <<?Statement    pipeline:hasParameter   ?Parameter>> pipeline:withParameterValue ?Parameter_value     .
        }
        
        # search for transformations in sklearn.preprocessing
        FILTER(REGEX(str(?Function_id), 'sklearn/preprocessing/'))
    
        # beautify output
        BIND(STRAFTER(str(?Function_id), str(preprocessing:)) as ?Transformation_1)
        BIND(REPLACE(?Transformation_1, '/fit_transform', '') as ?Transformation_2)
        BIND(REPLACE(?Transformation_2, '/inverse_transform', '') as ?Transformation_3)
        BIND(REPLACE(?Transformation_3, '/fit', '') as ?Transformation_4)
        BIND(REPLACE(?Transformation_4, '/transform', '') as ?Transformation)   
    }"""
    return execute_query(config, query)
