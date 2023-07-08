import os
import stardog as sd
from tqdm import tqdm

database = 'kgfarm_test'
os.system('unzip -o sample_data/graph/pipelines.zip -d sample_data/graph/')
pipelines_root = 'sample_data/graph/pipelines_rdf/'
data_items_root = 'sample_data/graph/data_items.ttl'
uri = 'http://kglids.org/resource/kaggle/{}/{}'


def time_taken(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def get_connection(database_name: str, drop_if_exists: bool):
    connection_details = \
    {
        'endpoint': 'http://localhost:5820',
        'username': 'admin',
        'password': 'admin'
    }

    with sd.Admin(**connection_details) as admin:
        if database_name in [db.name for db in admin.databases()]:
            print(database_name, 'exists!')
            if drop_if_exists:
                admin.database(database_name).drop()
                print(database_name, 'dropped!')
                _ = admin.new_database(database_name, {'edge.properties': True, 'search.enabled': True})
                print('new database: ', database_name, 'created!')

        else:
            print(database_name, ' does not exists!')
            _ = admin.new_database(database_name, {'edge.properties': True})
            print("new database: '", database_name, "' created!")

        conn = sd.Connection(database_name, **connection_details)
    return conn


def update_path_in_data_items_graph():
    with open(data_items_root, 'r') as file:
        lines = file.readlines()
    updated_lines = [line.replace('/Users/shubhamvashisth/Documents/data/data_lake/', f'{os.getcwd()}/sample_data/datasets/') for line in lines]

    with open(data_items_root, 'w') as file:
        file.writelines(updated_lines)
    print(os.getcwd())
    os.system(f'unzip -o {"sample_data/datasets.zip"} -d {"sample_data/"}')


def populate_pipelines_graph():
    os.system('find {} -name "*.DS_Store" -type f -delete'.format(pipelines_root))
    conn = get_connection(database, drop_if_exists=True)
    conn.begin()
    datasets = pipelines_root
    print('uploading pipelines from "{}" to stardog'.format(datasets))
    count = 0
    for dataset_name in tqdm(os.listdir(datasets)):
        if dataset_name == 'default.ttl' or dataset_name == 'library.ttl':
            os.system('stardog data add --format turtle {} {}'.format(database, pipelines_root+dataset_name))
            continue
        dataset = datasets + dataset_name + '/'
        for graph_name in os.listdir(dataset):
            graph = dataset + graph_name
            graph_uri = uri.format(dataset_name, graph_name[:graph_name.rindex('.')])
            conn.add(sd.content.File(graph), graph_uri=graph_uri)
            count = count + 1
    conn.commit()
    conn.close()
    print('Named graphs uploaded: ', count)


if __name__ == '__main__':
    populate_pipelines_graph()
    update_path_in_data_items_graph()
    os.system('stardog data add --format turtle {} {}'.format(database, data_items_root))
    os.chdir('../kg_augmentor/src/graph_builder')
    os.system(f'python build.py -db {database}')
    print(f'Setup success! Try running KGFarm_demo.ipynb')
