import os
import pandas as pd
import nbformat as nbf
from tqdm import tqdm
from operations.api import KGFarm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from helpers.helper import connect_to_stardog


class PipelineGenerator:
    def __init__(self, port: object = 5820, database: str = 'kgfarm_test', show_connection_status: bool = False):
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.notebook = nbf.v4.new_notebook()
        self.kgfarm = KGFarm()
        self.name = 'test1' + '.ipynb'
        self.cells = list()

    def __add(self, info: str, info_type: str = 'code'):
        if info_type == 'markdown':
            self.cells.append(nbf.v4.new_markdown_cell(info))
        elif info_type == 'code':
            self.cells.append(nbf.v4.new_code_cell(info))

    def __print_dataframe(self):
        code = """entity_df = pd.read_csv(entity_info.iloc[0].File_source)\nprint(entity_info.iloc[0].Physical_table)\nentity_df"""
        self.__add(code)

    def get_notebook_name(self):
        return self.name

    def write_to_notebook(self):
        self.notebook['cells'] = self.cells
        nbf.write(self.notebook, self.name)
        print('{} saved at {}'.format(self.name, os.getcwd()), '\nDone.')

    def instantiate_kgfarm(self):
        code = """from operations.api import KGFarm\nkgfarm = KGFarm()"""
        self.__add(code)

    def import_libraries(self):
        code = """import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.metrics import f1_score"""
        self.__add(code)

    def search_entity(self, entity_name: str = 'customer'):
        entity_info = self.kgfarm.search_entity(entity_name=entity_name)
        if len(entity_info):
            code = """entity_info = kgfarm.search_entity('{}')\nentity_info""".format(entity_name)
            self.__add(code)
            self.__print_dataframe()
        return entity_info

    def search_enrichment_options(self, entity_info: pd.DataFrame):
        entity_df = pd.read_csv(entity_info.iloc[0].File_source)
        enrichment_info = self.kgfarm.search_enrichment_options(entity_df=entity_df)
        if len(enrichment_info):
            code = """enrichment_info = kgfarm.search_enrichment_options(entity_df)\nenrichment_info"""
            self.__add(code)
            return enrichment_info, entity_df

    def enrich(self, enrich_info: tuple):
        entity_df = self.kgfarm.enrich(enrichment_info=enrich_info[0].iloc[0], entity_df=enrich_info[1])
        if len(entity_df):
            code = """entity_df = kgfarm.enrich(enrichment_info.iloc[0], entity_df, ttl=10)\nentity_df"""
            self.__add(code)
        return entity_df

    def search_transformations(self, entity_df: pd.DataFrame):
        transformation_info = self.kgfarm.recommend_feature_transformations(entity_df=entity_df)
        if len(transformation_info):
            code = """transformation_info = kgfarm.recommend_feature_transformations(entity_df)\ntransformation_info"""
            self.__add(code)
        return transformation_info, entity_df

    def apply_transformations(self, transformation_info: tuple):
        enrich_df = self.kgfarm.apply_transformation(transformation_info=transformation_info[0].iloc[0],
                                         entity_df=transformation_info[1])
        if len(enrich_df):
            code = """entity_df = kgfarm.apply_transformation(transformation_info.iloc[0], entity_df)\nentity_df"""
            self.__add(code)
        return enrich_df
        # add manual transformation for target

    def select_features(self, entity_df: pd.DataFrame, dependent_variable: str = 'membership'):
        X, y = self.kgfarm.select_features(entity_df=entity_df, dependent_variable=dependent_variable,
                                           plot_correlation=False,
                                           plot_anova_test=False,
                                           show_f_value=False, )
        if len(X) & len(y):
            code = """X, y = kgfarm.select_features(entity_df, dependent_variable='{}',\n""".format(dependent_variable) + \
                """plot_correlation=True, plot_anova_test=True, show_f_value=True)\nX"""
            self.__add(code)
        return X, y

    def split_data(self, data: tuple):
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.20, random_state=0)
        code = """X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)"""
        self.__add(code)
        return X_train, X_test, y_train, y_test

    def train_model(self, machine_learning_problem: str, data: tuple):
        X_train = data[0]
        X_test = data[1]
        y_train = data[2]
        y_test = data[3]

        if machine_learning_problem == 'classification':
            # instantiate the models
            random_forest_classifier = RandomForestClassifier()
            gradient_boosting_classifier = GradientBoostingClassifier()
            naive_bayes_classifier = GaussianNB()
            # fit the models on data
            random_forest_classifier.fit(X_train, y_train)
            gradient_boosting_classifier.fit(X_train, y_train)
            naive_bayes_classifier.fit(X_train, y_train)
            # add info to notebook
            models = """random_forest_classifier = RandomForestClassifier()\ngradient_boosting_classifier = GradientBoostingClassifier()\nnaive_bayes_classifier = GaussianNB()\n"""
            fit = """random_forest_classifier.fit(X_train, y_train)\ngradient_boosting_classifier.fit(X_train, y_train)\nnaive_bayes_classifier.fit(X_train, y_train)"""
            self.__add(models + fit)
            return random_forest_classifier, gradient_boosting_classifier, naive_bayes_classifier, X_test, y_test
        else:
            error = '{} not supported yet'.format(machine_learning_problem)
            raise ValueError(error)

    def evaluate_model(self):
        evaluate = """y_pred = random_forest_classifier.predict(X_test)\nf1_random_forest_classifier = round(f1_score(y_test, y_pred), 3)\ny_pred = gradient_boosting_classifier.predict(X_test)\nf1_gradient_boosting_classifier = round(f1_score(y_test, y_pred), 3)\ny_pred = naive_bayes_classifier.predict(X_test)\nf1_naive_bayes_classifier = round(f1_score(y_test, y_pred), 3)\n"""
        plot = """from helpers.helper import plot_scores\nscores = {'Random forest classifier': f1_random_forest_classifier,
                'Gradient boosting classifier': f1_gradient_boosting_classifier,
                'Naive bayes classifier': f1_naive_bayes_classifier}\nplot_scores(scores)\nfor classifier, f1 in conventional_approach.items():\n\tprint(f"{'{} (f1-score):'.format(classifier):<42}{f1:>1}")"""
        self.__add(evaluate + plot)


def run():
    pipeline_generator = PipelineGenerator()
    print('Generating {}...'.format(pipeline_generator.get_notebook_name()))
    status = [step for step in tqdm([pipeline_generator.instantiate_kgfarm(),
                     pipeline_generator.import_libraries(),
                     pipeline_generator.import_libraries(),
                     pipeline_generator.train_model(machine_learning_problem='classification',
                        data=pipeline_generator.split_data(
                            data=pipeline_generator.select_features(
                                entity_df=pipeline_generator.apply_transformations(
                                    transformation_info=pipeline_generator.search_transformations(
                                        entity_df=pipeline_generator.enrich(
                                            enrich_info=pipeline_generator.search_enrichment_options(
                                                entity_info=pipeline_generator.search_entity()))))))),
                     pipeline_generator.evaluate_model(),
                     pipeline_generator.write_to_notebook()])]


if __name__ == "__main__":
    run()
