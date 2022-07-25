# The Entities & Feature Views were predicted using Discovery operations of KGFarm. Feel free to edit :)
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
branch_database = Entity(name='branch_database', value_type=ValueType.INT64, join_key='Branch Number')

cfo_to_assets_data = Entity(name='cfo_to_assets_data', value_type=ValueType.INT64, join_key=' CFO to Assets')

country_code_EdStatsCountry = Entity(name='country_code_EdStatsCountry', value_type=ValueType.STRING, join_key='Country Code')

country_name_API_ILO_country_YU = Entity(name='country_name_API_ILO_country_YU', value_type=ValueType.STRING, join_key='Country Name')

country_name_country_population = Entity(name='country_name_country_population', value_type=ValueType.STRING, join_key='Country Name')

country_name_fertility_rate = Entity(name='country_name_fertility_rate', value_type=ValueType.STRING, join_key='Country Name')

country_name_life_expectancy = Entity(name='country_name_life_expectancy', value_type=ValueType.STRING, join_key='Country Name')

entity_udemy_output_All_Finance_Accounting_p1_p626 = Entity(name='entity_udemy_output_All_Finance_Accounting_p1_p626', value_type=ValueType.INT64, join_key='id')

franch_TeamsFranchises = Entity(name='franch_TeamsFranchises', value_type=ValueType.STRING, join_key='franchID')

indicator_name_EdStatsSeries = Entity(name='indicator_name_EdStatsSeries', value_type=ValueType.STRING, join_key='Indicator Name')

inflation_annual_cpi_african_crises = Entity(name='inflation_annual_cpi_african_crises', value_type=ValueType.INT64, join_key='inflation_annual_cpi')

loan_ibrd_statement_of_loans_latest_available_snapshot = Entity(name='loan_ibrd_statement_of_loans_latest_available_snapshot', value_type=ValueType.STRING, join_key='Loan Number')

player_Master = Entity(name='player_Master', value_type=ValueType.STRING, join_key='playerID')

row_ChurnModeling = Entity(name='row_ChurnModeling', value_type=ValueType.INT64, join_key='RowNumber')

row_Churn_Modelling = Entity(name='row_Churn_Modelling', value_type=ValueType.INT64, join_key='RowNumber')

row_churn = Entity(name='row_churn', value_type=ValueType.INT64, join_key='RowNumber')

short_name_Country = Entity(name='short_name_Country', value_type=ValueType.STRING, join_key='Short.Name')

total_deposits_banks = Entity(name='total_deposits_banks', value_type=ValueType.INT64, join_key='Total Deposits')

variance_BankNote_Authentication = Entity(name='variance_BankNote_Authentication', value_type=ValueType.INT64, join_key='variance')

Feature_view_04 = FeatureView(
name='Feature_view_04',
entities=['branch_database'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/chasebank.bank-deposits/data/database.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_05 = FeatureView(
name='Feature_view_05',
entities=['total_deposits_banks'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/fdic.bank-failures/data/banks.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_07 = FeatureView(
name='Feature_view_07',
entities=['inflation_annual_cpi_african_crises'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/chirin.africa-economic-banking-and-systemic-crisis-data/data/african_crises.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_15 = FeatureView(
name='Feature_view_15',
entities=['entity_udemy_output_All_Finance_Accounting_p1_p626'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/jilkothari.finance-accounting-courses-udemy-13k-course/data/udemy_output_All_Finance__Accounting_p1_p626.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_16 = FeatureView(
name='Feature_view_16',
entities=['country_name_fertility_rate'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/gemartin.world-bank-data-1960-to-2016/data/fertility_rate.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_19 = FeatureView(
name='Feature_view_19',
entities=['player_Master'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/Master.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_21 = FeatureView(
name='Feature_view_21',
entities=['country_name_country_population'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/gemartin.world-bank-data-1960-to-2016/data/country_population.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_22 = FeatureView(
name='Feature_view_22',
entities=['row_Churn_Modelling'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/barelydedicated.bank-customer-churn-modeling/data/Churn_Modelling.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_23 = FeatureView(
name='Feature_view_23',
entities=['cfo_to_assets_data'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/fedesoriano.company-bankruptcy-prediction/data/data.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_24 = FeatureView(
name='Feature_view_24',
entities=['row_Churn_Modelling'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/adammaus.predicting-churn-for-bank-customers/data/Churn_Modelling.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_33 = FeatureView(
name='Feature_view_33',
entities=['row_churn'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/mathchi.churn-for-bank-customers/data/churn.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_34 = FeatureView(
name='Feature_view_34',
entities=['row_ChurnModeling'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/santoshd3.bank-customers/data/Churn Modeling.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_36 = FeatureView(
name='Feature_view_36',
entities=['short_name_Country'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.world-gender-statistics/data/Country.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_37 = FeatureView(
name='Feature_view_37',
entities=['country_code_EdStatsCountry'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.education-statistics/data/EdStatsCountry.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_39 = FeatureView(
name='Feature_view_39',
entities=['variance_BankNote_Authentication'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/ritesaluja.bank-note-authentication-uci-data/data/BankNote_Authentication.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_41 = FeatureView(
name='Feature_view_41',
entities=['loan_ibrd_statement_of_loans_latest_available_snapshot'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.ibrd-statement-of-loans-data/data/ibrd-statement-of-loans-latest-available-snapshot.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_47 = FeatureView(
name='Feature_view_47',
entities=['franch_TeamsFranchises'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/TeamsFranchises.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_48 = FeatureView(
name='Feature_view_48',
entities=['indicator_name_EdStatsSeries'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.education-statistics/data/EdStatsSeries.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_51 = FeatureView(
name='Feature_view_51',
entities=['country_name_life_expectancy'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/gemartin.world-bank-data-1960-to-2016/data/life_expectancy.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_52 = FeatureView(
name='Feature_view_52',
entities=['country_name_API_ILO_country_YU'],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/sovannt.world-bank-youth-unemployment/data/API_ILO_country_YU.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_01 = FeatureView(
name='Feature_view_01',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/Pitching.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_02 = FeatureView(
name='Feature_view_02',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/AllstarFull.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_03 = FeatureView(
name='Feature_view_03',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/ManagersHalf.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_06 = FeatureView(
name='Feature_view_06',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.world-gender-statistics/data/FootNote.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_08 = FeatureView(
name='Feature_view_08',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/Batting.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_09 = FeatureView(
name='Feature_view_09',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/Fielding.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_10 = FeatureView(
name='Feature_view_10',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/Teams.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_11 = FeatureView(
name='Feature_view_11',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/Salaries.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_12 = FeatureView(
name='Feature_view_12',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/Managers.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_13 = FeatureView(
name='Feature_view_13',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.education-statistics/data/EdStatsFootNote.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_14 = FeatureView(
name='Feature_view_14',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/TeamsHalf.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_17 = FeatureView(
name='Feature_view_17',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.world-gender-statistics/data/Country-Series.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_18 = FeatureView(
name='Feature_view_18',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.world-gender-statistics/data/Data.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_20 = FeatureView(
name='Feature_view_20',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/PitchingPost.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_25 = FeatureView(
name='Feature_view_25',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/BattingPost.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_26 = FeatureView(
name='Feature_view_26',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/SeriesPost.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_27 = FeatureView(
name='Feature_view_27',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/AwardsPlayers.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_28 = FeatureView(
name='Feature_view_28',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/FieldingOF.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_29 = FeatureView(
name='Feature_view_29',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/ealaxi.banksim1/data/bs140513_032310.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_30 = FeatureView(
name='Feature_view_30',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/janiobachmann.bank-marketing-dataset/data/bank.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_31 = FeatureView(
name='Feature_view_31',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/AwardsSharePlayers.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_32 = FeatureView(
name='Feature_view_32',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/AwardsShareManagers.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_35 = FeatureView(
name='Feature_view_35',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/HallOfFame.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_38 = FeatureView(
name='Feature_view_38',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/ealaxi.banksim1/data/bsNET140513_032310.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_40 = FeatureView(
name='Feature_view_40',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.ibrd-statement-of-loans-data/data/ibrd-statement-of-loans-historical-data.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_42 = FeatureView(
name='Feature_view_42',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.education-statistics/data/EdStatsData.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_43 = FeatureView(
name='Feature_view_43',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/open-source-sports.baseball-databank/data/AwardsManagers.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_44 = FeatureView(
name='Feature_view_44',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/kaggle.us-consumer-finance-complaints/data/consumer_complaints.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_45 = FeatureView(
name='Feature_view_45',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.education-statistics/data/EdStatsCountry-Series.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_46 = FeatureView(
name='Feature_view_46',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.world-gender-statistics/data/Series.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_49 = FeatureView(
name='Feature_view_49',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.health-nutrition-and-population-statistics/data/data.parquet',
event_timestamp_column='event_timestamp')
)

Feature_view_50 = FeatureView(
name='Feature_view_50',
entities=[],
ttl=timedelta(days=100),
online=True,
batch_source=FileSource(
path=r'/Users/shubhamvashisth/Documents/data/kaggle/parquet/theworldbank.world-gender-statistics/data/Series-Time.parquet',
event_timestamp_column='event_timestamp')
)
