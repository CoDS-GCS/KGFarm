# The Entities & Feature Views were predicted using Discovery operations. Feel free to edit :)
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

client_id = Entity(name='client_id', value_type=ValueType.INT64, join_key='client_id')

account_id = Entity(name='account_id', value_type=ValueType.STRING, join_key='account_id')

feature_view_1 = FeatureView(
name='feature_view_1',
entities=['district_id'],
ttl=timedelta(days=1000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completeddistrict.parquet',
event_timestamp_column='event_timestamp')
)

feature_view_2 = FeatureView(
name='feature_view_2',
entities=['client_id'],
ttl=timedelta(days=1000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completeddisposition.parquet',
event_timestamp_column='event_timestamp')
)

feature_view_3 = FeatureView(
name='feature_view_3',
entities=['account_id'],
ttl=timedelta(days=1000),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedacct.parquet',
event_timestamp_column='event_timestamp')
)
