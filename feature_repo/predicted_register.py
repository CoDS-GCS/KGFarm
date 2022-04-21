'''
The Entities & Feature Views were predicted using Discovery operations. Feel free to edit :)
'''
from datetime import timedelta
from feast import Entity, FeatureView, FileSource, ValueType

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

district_id = Entity(name='district_id', value_type=ValueType.INT64, join_key='district_id')

feature_view_1 = FeatureView(
name='predicted_feature_view_1',
entities=['district_id'],
ttl=timedelta(weeks=1000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/data_discovery/helpers/sample_data/parquet/Retail Banking-demo-data/crm_reviews.parquet',
event_timestamp_column='timestamp')
)

account_id = Entity(name='account_id', value_type=ValueType.STRING, join_key='account_id')

feature_view_2 = FeatureView(
name='predicted_feature_view_2',
entities=['account_id'],
ttl=timedelta(weeks=1000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/data_discovery/helpers/sample_data/parquet/Retail Banking-demo-data/completeddisposition.parquet',
event_timestamp_column='timestamp')
)

feature_view_3 = FeatureView(
name='predicted_feature_view_3',
entities=['account_id'],
ttl=timedelta(weeks=1000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/data_discovery/helpers/sample_data/parquet/Retail Banking-demo-data/completedorder.parquet',
event_timestamp_column='timestamp')
)
