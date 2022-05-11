def definitions():
    return \
"""# The Entities & Feature Views were predicted using Discovery operations. Feel free to edit :)
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
'''"""


def entity_skeleton():
    return \
"""
{} = Entity(name='{}', value_type=ValueType.{}, join_key='{}')
"""


def feature_view_skeleton():
    return \
        """
{} = FeatureView(
name='{}',
entities=['{}'],
ttl=timedelta(days={}),
online=True,
batch_source=FileSource(
path=r'{}',
event_timestamp_column='event_timestamp')
)
"""
