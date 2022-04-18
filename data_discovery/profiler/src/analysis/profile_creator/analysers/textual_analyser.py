from analysis.profile_creator.analysers.i_analyser import IAnalyser
from datasketch import MinHash
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, sum, collect_list, udf
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField


class TextualAnalyser(IAnalyser):

    def __init__(self, df: DataFrame):
        self.profiles_info = {}
        self.df = df

    def get_profiles_info(self):
        return self.profiles_info

    def analyse_columns(self):
        self.profiles_info = {}
        columns = self.df.columns
        if not columns:
            return
        minhash_distinct_values_count_dict = self.__get_minhash_distinct_values_counts(columns)
        missing_values_dict = self.__get_missing_values(self.df.columns)
        for column in columns:
            info = minhash_distinct_values_count_dict[column]
            info.update({'missing_values_count': missing_values_dict[column]})
            self.profiles_info[column] = info

    def __get_minhash_distinct_values_counts(self, columns: list) -> dict:
        def compute_minhash(l):
            m = MinHash(num_perm=512)
            for v in l:
                if isinstance(v, str):
                    m.update(v.lower().encode('utf8'))
            return m.hashvalues.tolist(), len(l), len(set(l))

        if not columns:
            return columns
        
        '''profiles_info = self.df.rdd \
            .map(lambda row: row.asDict()) \
            .flatMap(lambda d: [(c, d[c]) for c in columns]) \
            .groupByKey() \
            .map(lambda column: {column[0]:
                                     {'minhash': compute_minhash(column[1]), 'count': len(column[1]),
                                      'distinct_values_count': len(set(column[1]))}}) \
            .reduce(lambda x, y: {**x, **y})
        return profiles_info'''
        schema = StructType([StructField('minhash', ArrayType(IntegerType()), False),
                 StructField('count', IntegerType(), False),
                 StructField('distinct', IntegerType(), False)])
        minhashUDF = udf(lambda z: compute_minhash(z), schema)
        #minhashUDF = udf(lambda z: compute_minhash(z))
        cols = self.df.columns
        cols2 = ['`' + c + '`' for c in cols]
        df2 = self.df.select([collect_list(c) for c in cols2]).toDF(*cols2)
        df2 = df2.toDF(*cols)
        for col in cols:
                df2 = df2.withColumn(col, minhashUDF('`' + col + '`'))
        profiles_info = {}
        d = df2.toPandas().to_dict()
        for c in cols:
                #col_rdd = df.select('`' + c + '`').rdd
                profiles_info[c] = {'minhash': d[c][0]['minhash'], 'count':  d[c][0]['count'],
                'distinct_values_count' :  d[c][0]['distinct']}
        return profiles_info


    def __get_missing_values(self, columns: list) -> dict:
        return self.df.select(*(sum(col('`' + c + '`').isNull().cast("int")).alias(c) for c in columns)) \
            .rdd \
            .map(lambda x: x.asDict()) \
            .collect()[0]

