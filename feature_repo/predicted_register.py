# The Entities & Feature Views were predicted using Discovery operations. Feel free to edit :)
from datetime import timedelta
from feast import Entity, FeatureView, FileSource, ValueType
'''
####################################################################################################################
Entity: 
        An entity is the object on which features are observed.
        Used when joining feature values in order to build one large data set.
--------------------------------------------------------------------------------------------------------------------
Feature View: 
        A feature view is an object that represents a logical group of feature data as it is found in a data source.
####################################################################################################################
'''
account_completedacct = Entity(name='account_completedacct', value_type=ValueType.STRING, join_key='account_id')

account_completedloan = Entity(name='account_completedloan', value_type=ValueType.STRING, join_key='account_id')

client_completedclient = Entity(name='client_completedclient', value_type=ValueType.STRING, join_key='email')

client_completeddisposition = Entity(name='client_completeddisposition', value_type=ValueType.INT64, join_key='client_id')

disp_completedcard = Entity(name='disp_completedcard', value_type=ValueType.STRING, join_key='disp_id')

district_completeddistrict = Entity(name='district_completeddistrict', value_type=ValueType.INT64, join_key='district_id')

loan_luxuryloanportfolio = Entity(name='loan_luxuryloanportfolio', value_type=ValueType.STRING, join_key='loan_id')

order_completedorder = Entity(name='order_completedorder', value_type=ValueType.INT64, join_key='order_id')

account_month_summary = Entity(name='account_month_summary', value_type=ValueType.STRING, join_key='account_id')

client_month_summary = Entity(name='client_month_summary', value_type=ValueType.INT64, join_key='client_id')

Feature_view_7 = FeatureView(
name='Feature_view_7',
entities=['account_completedacct'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedacct.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_9 = FeatureView(
name='Feature_view_9',
entities=['district_completeddistrict'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completeddistrict.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_13 = FeatureView(
name='Feature_view_13',
entities=['disp_completedcard'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedcard.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_11 = FeatureView(
name='Feature_view_11',
entities=['account_month_summary', 'client_month_summary'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/month_summary.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_5 = FeatureView(
name='Feature_view_5',
entities=['account_completedloan'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedloan.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_10 = FeatureView(
name='Feature_view_10',
entities=['client_completeddisposition'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completeddisposition.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_8 = FeatureView(
name='Feature_view_8',
entities=['order_completedorder'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedorder.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_1 = FeatureView(
name='Feature_view_1',
entities=['client_completedclient'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedclient.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_3 = FeatureView(
name='Feature_view_3',
entities=['loan_luxuryloanportfolio'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/luxuryloanportfolio.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_14 = FeatureView(
name='Feature_view_14',
entities=[],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/district.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_12 = FeatureView(
name='Feature_view_12',
entities=[],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/account.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_6 = FeatureView(
name='Feature_view_6',
entities=[],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/crm_reviews.parquet',
event_timestamp_column='event_timestamp')
)
