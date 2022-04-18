class DataProfile:

    def __init__(self, pid: float, origin: str, dataset_name: str, path: str, table_name: str, column_name: str,
                 data_type: str,
                 total_values: float, distinct_values_count: float, missing_values_count: float, min_value: float,
                 max_value: float, mean: float, median: float, iqr: float,
                 minhash: list):
        self.pid = pid
        self.origin = origin
        self.dataset_name = dataset_name
        self.path = path
        self.table_name = table_name
        self.column_name = column_name
        self.data_type = data_type
        self.total_values = total_values
        self.distinct_values_count = distinct_values_count
        self.missing_values_count = missing_values_count
        self.max_value = max_value
        self.min_value = min_value
        self.mean = mean
        self.median = median
        self.iqr = iqr
        self.minhash = minhash

    def get_pid(self) -> float:
        return self.pid

    def get_origin(self) -> str:
        return self.origin

    def get_dataset_name(self) -> str:
        return self.dataset_name

    def get_path(self) -> str:
        return self.path

    def get_table_name(self) -> str:
        return self.table_name

    def get_column_name(self) -> str:
        return self.column_name

    def get_data_type(self) -> str:
        return self.data_type

    def get_total_values(self) -> float:
        return self.total_values

    def get_distinct_values_count(self) -> float:
        return self.distinct_values_count

    def get_missing_values_count(self) -> float:
        return self.missing_values_count

    def get_minhash(self) -> list:
        return self.minhash

    def get_min_value(self) -> float:
        return self.min_value

    def get_max_value(self) -> float:
        return self.max_value

    def get_mean(self) -> float:
        return self.mean

    def get_median(self) -> float:
        return self.median

    def get_iqr(self) -> float:
        return self.iqr

    def set_pid(self, pid: float):
        self.pid = pid

    def set_origin(self, origin: str):
        self.origin = origin

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

    def set_path(self, path: str):
        self.path = path

    def set_table_name(self, table_name: str):
        self.table_name = table_name

    def set_column_name(self, column_name: str):
        self.column_name = column_name

    def set_data_type(self, data_type: str):
        self.data_type = data_type

    def set_total_values(self, total_values: float):
        self.total_values = total_values

    def set_distinct_values_count(self, unique_values: float):
        self.distinct_values_count = unique_values

    def set_missing_values_count(self, missing_values_count: float):
        self.missing_values_count = missing_values_count

    def set_min_value(self, min_value: float):
        self.min_value = min_value

    def set_max_value(self, max_value: float):
        self.max_value = max_value

    def set_mean(self, mean: float):
        self.mean = mean

    def set_median(self, median: float):
        self.median = median

    def set_iqr(self, iqr: float):
        self.iqr = iqr

    def set_minhash(self, minhash: list):
        self.minhash = minhash

    def __str__(self):
        return self.table_name + ': ' + str(self.minhash)
