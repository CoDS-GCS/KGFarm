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
account__completedacct = Entity(name='account__completedacct', value_type=ValueType.STRING, join_key='account_id')

card__completedcard = Entity(name='card__completedcard', value_type=ValueType.STRING, join_key='card_id')

loan__luxuryloanportfolio = Entity(name='loan__luxuryloanportfolio', value_type=ValueType.STRING, join_key='loan_id')

account__completedloan = Entity(name='account__completedloan', value_type=ValueType.STRING, join_key='account_id')

district__completeddistrict = Entity(name='district__completeddistrict', value_type=ValueType.INT64, join_key='district_id')

client__completeddisposition = Entity(name='client__completeddisposition', value_type=ValueType.INT64, join_key='client_id')

order__completedorder = Entity(name='order__completedorder', value_type=ValueType.INT64, join_key='order_id')

client__completedclient = Entity(name='client__completedclient', value_type=ValueType.STRING, join_key='client_id')

account__month_summary = Entity(name='account__month_summary', value_type=ValueType.STRING, join_key='account_id')

client__month_summary = Entity(name='client__month_summary', value_type=ValueType.INT64, join_key='client_id')

Feature_view_1 = FeatureView(
name='Feature_view_1',
entities=['loan__luxuryloanportfolio'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/luxuryloanportfolio.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_10 = FeatureView(
name='Feature_view_10',
entities=['card__completedcard'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedcard.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_13 = FeatureView(
name='Feature_view_13',
entities=['account__month_summary', 'client__month_summary'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/month_summary.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_2 = FeatureView(
name='Feature_view_2',
entities=['account__completedloan'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedloan.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_3 = FeatureView(
name='Feature_view_3',
entities=['account__completedacct'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedacct.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_4 = FeatureView(
name='Feature_view_4',
entities=['client__completedclient'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedclient.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_6 = FeatureView(
name='Feature_view_6',
entities=['district__completeddistrict'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completeddistrict.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_7 = FeatureView(
name='Feature_view_7',
entities=['order__completedorder'],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/completedorder.parquet',
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

Feature_view_11 = FeatureView(
name='Feature_view_11',
entities=[],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/district.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_5 = FeatureView(
name='Feature_view_5',
entities=[],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/crm_call_center_logs.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_14 = FeatureView(
name='Feature_view_14',
entities=[],
ttl=timedelta(days=30),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/CoDS/projects/KGFarm/helpers/sample_data/parquet/retail-bankingdemodata/client.parquet',
event_timestamp_column='event_timestamp')
)
