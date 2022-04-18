from analysis.profile_creator.analysers.numerical_analyser import NumericalAnalyser
from analysis.profile_creator.analysers.textual_analyser import TextualAnalyser
from data.data_profile import DataProfile
from data.tables.i_table import ITable
from pyspark.sql import DataFrame
from utils import generate_id


class ProfileCreator:

    def __init__(self, table: ITable):
        self.table = table

    def create_numerical_profiles(self, numerical_cols_df: DataFrame):
        numerical_analyser = NumericalAnalyser(numerical_cols_df)
        numerical_analyser.analyse_columns()
        profiles_info = numerical_analyser.get_profiles_info()
        dataset_name = self.table.get_dataset_name()
        table_name = self.table.get_table_name()
        path = self.table.get_table_path()
        origin = self.table.get_origin()
        for column_name in numerical_cols_df.columns:
            if column_name == '__index_level_0__':
                continue
            pid = generate_id(dataset_name, table_name, column_name)
            profile_info = profiles_info[column_name]
            profile = DataProfile(pid, origin, dataset_name, path, table_name, column_name, 'N',
                                  float(profile_info['count']),
                                  float(profile_info['distinct_values_count']),
                                  float(profile_info['missing_values_count']),
                                  float(profile_info['min']), float(profile_info['max']),
                                  float(profile_info['mean']), float(profile_info['50%']),
                                  float(profile_info['75%']) - float(profile_info['25%']), [])
            yield profile

    def create_textual_profiles(self, textual_cols_df: DataFrame):
        textual_analyser = TextualAnalyser(textual_cols_df)
        textual_analyser.analyse_columns()
        profiles_info = textual_analyser.get_profiles_info()
        dataset_name = self.table.get_dataset_name()
        table_name = self.table.get_table_name()
        path = self.table.get_table_path()
        origin = self.table.get_origin()
        for column_name in textual_cols_df.columns:
            if column_name == '__index_level_0__':
                continue
            pid = generate_id(dataset_name, table_name, column_name)
            profile_info = profiles_info[column_name]
            profile = DataProfile(pid, origin, dataset_name, path, table_name, column_name, 'T',
                                  float(profile_info['count']),
                                  float(profile_info['distinct_values_count']),
                                  float(profile_info['missing_values_count']),
                                  -1, -1, -1, -1, -1,
                                  profile_info['minhash'])
            yield profile
