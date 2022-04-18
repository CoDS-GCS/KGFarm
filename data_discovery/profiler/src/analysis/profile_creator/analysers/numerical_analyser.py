                                                            
from analysis.profile_creator.analysers.i_analyser import IAnalyser
from analysis.utils import init_spark
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, countDistinct, sum


class NumericalAnalyser(IAnalyser):

    def __init__(self, df: DataFrame):
        self.spark = init_spark()
        self.profiles_info = {}
        self.df = df

    def analyse_columns(self):
        columns = self.df.columns
        if not columns:
            return
        summaries = self.__extract_summaries(columns)
        num_distinct_values_counts_per_column_dict = self.__compute_distinct_values_counts(columns)
        missing_values_per_column_dict = self.__get_missing_values(columns)
        quartiles = self.__get_quantiles(columns)
        for col in columns:
            profile_info = {**{'count': summaries[col][0]},
                            **{'mean': summaries[col][1]},
                            **{'stddev': summaries[col][2]},
                            **{'min': summaries[col][3]},
                            **{'max': summaries[col][4]},
                            **{'distinct_values_count': num_distinct_values_counts_per_column_dict[col][0]},
                            **{'missing_values_count': missing_values_per_column_dict[col]},
                            **{'missing_values_count': 0},
                            **{'25%': quartiles['`' +col+'`' ][0]},
                            **{'50%': quartiles['`' +col+'`' ][1]},
                            **{'75%': quartiles['`' +col+'`' ][2]}}
            self.profiles_info[col] = profile_info

    def __extract_summaries(self, columns: list):
        summary = self.df.describe()
        return summary.toPandas().to_dict()

    def __compute_distinct_values_counts(self, columns: list) -> dict:
        return self.df.agg(*(countDistinct(col('`' + c + '`')).alias(c) for c in columns)).toPandas().to_dict()

    def __get_missing_values(self, columns: list) -> dict:
        return self.df.select(*(sum(col('`' + c + '`').isNull().cast("int")).alias(c) for c in columns)) \
            .rdd \
            .map(lambda x: x.asDict()) \
            .collect()[0]

    def __get_quantiles(self, quantile_list) -> dict:
        quantile_list = ['`' + c + '`' for c in quantile_list]
        quartiles = [0.25, 0.5, 0.75]
        quartilesDF = self.spark.createDataFrame(
            zip(quartiles, *self.df.approxQuantile(quantile_list, quartiles, 0.03)),
            ["Percentile"] + quantile_list
        )
        return quartilesDF.toPandas().to_dict()

            
    def get_profiles_info(self):
        return self.profiles_info
