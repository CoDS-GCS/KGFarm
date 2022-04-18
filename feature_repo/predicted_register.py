'''
The Entities & Feature Views were predicted using Discovery operations. Feel free to edit :)
'''
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType

'''
####################################################################################################################
Entity: 
        An entity is the object on which features are observed.
        Used when joining feature values n order to build one large data set.
--------------------------------------------------------------------------------------------------------------------
Feature View: 
        A feature view is an object that represents a logical group of feature data as it is found in a data source.
####################################################################################################################
'''

entity_1 = Entity(name='account_id', value_type=ValueType.STRING, join_key='account_id')

feature_view_1 = FeatureView(
name='predicted_feature_view_1',
entities=['account_id'],
ttl=timedelta(weeks=10000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/borealisAI/projects/data_discovery/sample_data/parquet/Retail Banking-demo-data/completeddisposition.parquet',
event_timestamp_column='timestamp')
)

entity_2 = Entity(name='district_id', value_type=ValueType.INT64, join_key='district_id')

feature_view_2 = FeatureView(
name='predicted_feature_view_2',
entities=['district_id'],
ttl=timedelta(weeks=10000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/borealisAI/projects/data_discovery/sample_data/parquet/Retail Banking-demo-data/crm_reviews.parquet',
event_timestamp_column='timestamp')
)

entity_3 = Entity(name='district_id', value_type=ValueType.INT64, join_key='district_id')

feature_view_3 = FeatureView(
name='predicted_feature_view_3',
entities=['district_id'],
ttl=timedelta(weeks=10000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/borealisAI/projects/data_discovery/sample_data/parquet/Retail Banking-demo-data/completeddistrict.parquet',
event_timestamp_column='timestamp')
)

entity_4 = Entity(name='account_id', value_type=ValueType.STRING, join_key='account_id')

feature_view_4 = FeatureView(
name='predicted_feature_view_4',
entities=['account_id'],
ttl=timedelta(weeks=10000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/borealisAI/projects/data_discovery/sample_data/parquet/Retail Banking-demo-data/completedacct.parquet',
event_timestamp_column='timestamp')
)

entity_5 = Entity(name='account_id', value_type=ValueType.STRING, join_key='account_id')

feature_view_5 = FeatureView(
name='predicted_feature_view_5',
entities=['account_id'],
ttl=timedelta(weeks=10000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/borealisAI/projects/data_discovery/sample_data/parquet/Retail Banking-demo-data/completedorder.parquet',
event_timestamp_column='timestamp')
)
