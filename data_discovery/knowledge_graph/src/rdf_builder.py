import uuid
from collections import defaultdict
from math import isinf

import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.cluster import DBSCAN

from enums.relation import Relation
from rdf_resource import RDFResource
from triplet import Triplet
from utils import generate_label
from word_embedding.embeddings_client import get_similarity_between
from word_embedding.word_embeddings_services import WordEmbeddingServices


def _generate_id(name, dic):
    if name in dic.keys():
        return dic[name]
    else:
        return str(uuid.uuid1().int)


class RDFBuilder:
    global lac
    lac = 'lac'

    def __init__(self):
        self.word_embeddings = WordEmbeddingServices()
        self.__triplets = []
        self.__triplets_to_dump = []
        self.__docs = []
        print('** Building the rdf has started **')
        self.__namespaces = {'lac': 'http://www.example.com/lac#',
                             'schema': 'http://schema.org/',
                             'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                             'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
                             'owl': 'http://www.w3.org/2002/07/owl#',
                             'dct': 'http://purl.org/dc/terms/'}
        self._column_id_to_name = dict()
        self._table_id_to_name = dict()
        self._table_id_to_origin = dict()
        self._dataset_id_to_name = dict()
        self.__source_ids = defaultdict(list)
        self.__neighbors = defaultdict(dict)
        self.__id_to_labels = dict()

    def initialize_nodes(self, fields):
        def create_triplets(rdf_source, predicate, objct):
            triplet = Triplet(rdf_source,
                              predicate,
                              RDFResource(objct))
            self.__triplets.append(triplet)

        def create_predicate(relation, namespace):
            return RDFResource(relation, False, self.__namespaces[namespace])

        def _generate_table_id(name, dic):
            return _generate_id(name, dic)

        def _generate_dataset_id(name, dic):
            return _generate_id(name, dic)

        print('** ** Initializing the nodes has started ** **')
        for (
                column_id, origin, dataset_name, table_name, column_name, total_values_count, distinct_values_count,
                missing_values_count, data_type, median, minValue, maxValue, path) in fields:
            self._column_id_to_name[column_id] = (dataset_name, table_name, column_name, data_type)
            table_id = _generate_table_id(table_name, self._table_id_to_name)
            dataset_id = _generate_dataset_id(dataset_name, self._dataset_id_to_name)
            self.__source_ids[table_name].append(column_id)
            self.__docs.append(column_name)

            column_node = RDFResource(column_id, False, self.__namespaces[lac])
            table_node = RDFResource(table_id, False, self.__namespaces[lac])
            dataset_node = RDFResource(dataset_id, False, self.__namespaces[lac])
            col_label = generate_label(column_name, 'en')
            self.__id_to_labels[column_id] = col_label.get_text()

            # Create the meta data triplet (data_type, table_name, and column_name)
            create_triplets(column_node, create_predicate('type', 'schema'), data_type)
            create_triplets(column_node, create_predicate('name', 'schema'), column_name)
            create_triplets(column_node, create_predicate('totalVCount', 'schema'), total_values_count)
            create_triplets(column_node, create_predicate('distinctVCount', 'schema'), distinct_values_count)
            create_triplets(column_node, create_predicate('missingVCount', 'schema'), missing_values_count)
            create_triplets(column_node, create_predicate('origin', 'lac'), origin)
            create_triplets(column_node, create_predicate('label', 'rdfs'), col_label)
            if data_type == 'N':
                create_triplets(column_node, create_predicate('median', 'schema'), median)
                create_triplets(column_node, create_predicate('maxValue', 'schema'), maxValue)
                create_triplets(column_node, create_predicate('minValue', 'schema'), minValue)

            create_triplets(column_node, create_predicate(Relation.isPartOf.name, 'dct'), table_node)
            create_triplets(column_node, create_predicate('type', 'rdf'),
                            RDFResource('column', False, self.__namespaces[lac]))

            if table_id not in self._table_id_to_origin.keys():
                self._table_id_to_origin[table_id] = origin

            if not (table_name in self._table_id_to_name.keys() and dataset_name in self._dataset_id_to_name.keys()):
                table_label = generate_label(table_name, 'en')
                create_triplets(table_node, create_predicate('name', 'schema'), table_name)
                create_triplets(table_node, create_predicate('label', 'rdfs'), table_label)
                create_triplets(table_node, create_predicate('path', 'lac'), path)
                create_triplets(table_node, create_predicate(Relation.isPartOf.name, 'dct'), dataset_node)
                create_triplets(table_node, create_predicate('type', 'rdf'),
                                RDFResource('table', False, self.__namespaces[lac]))
                if dataset_name not in self._dataset_id_to_name.keys():
                    dataset_label = generate_label(dataset_name, 'en')
                    create_triplets(dataset_node, create_predicate('name', 'schema'), dataset_name)
                    create_triplets(dataset_node, create_predicate('label', 'rdfs'), dataset_label)
                    create_triplets(dataset_node, create_predicate('type', 'rdf'),
                                    RDFResource('dataset', False, self.__namespaces[lac]))
                    self._dataset_id_to_name[dataset_name] = dataset_id
                self._table_id_to_name[table_name] = table_id

            cardinality_ratio = None
            if float(total_values_count) > 0:
                cardinality_ratio = float(distinct_values_count) / float(total_values_count)
                create_triplets(column_node, create_predicate(Relation.cardinality.name, 'owl'), cardinality_ratio)
                self.__neighbors[column_id].update({'cardinality': cardinality_ratio})

            # append origins to tables
            for table_id in self._table_id_to_origin.keys():
                table_node = RDFResource(table_id, False, self.__namespaces[lac])
                create_triplets(table_node,
                                create_predicate('origin', lac),
                                self._table_id_to_origin[table_id])
        self.__triplets_to_dump.extend(self.__triplets)
        print('** ** Initializing the nodes has ended ** **')

    def neighbors_id(self, arg, relation):
        if isinstance(arg, str):
            nid = arg
        elif isinstance(arg, float):
            nid = arg
        elif isinstance(arg, int):
            nid = arg
        else:
            raise ValueError('arg must be either str, int, or float')
        nid = str(nid)
        data = []
        if not relation.name in self.__neighbors[nid]:
            return data
        neighbours = self.__neighbors[nid][relation.name]
        for neighbor_id, score in neighbours:
            (db_name, source_name, field_name, data_type) = self._column_id_to_name[neighbor_id]
            data.append((neighbor_id, db_name, source_name, field_name, score))
        return data

    def get_triplets(self):
        return list(set([str(t) for t in self.__triplets]))

    def dump_into_file(self, filename):
        with open(filename, 'w') as f:
            for item in self.__triplets_to_dump:
                f.write("%s\n" % item)

    def set_neighbors(self, relation, nid1, nid2, score):
        if not relation in self.__neighbors[nid1]:
            self.__neighbors[nid1].update({relation: [(nid2, score)]})
        else:
            if not (nid2, score) in self.__neighbors[nid1][relation]:
                self.__neighbors[nid1][relation].append((nid2, score))

    def build_semantic_sim_relation(self):

        def create_triplets(nid1, nid2, score):
            nestedSubject = RDFResource(nid1, False, self.__namespaces[lac])
            nestedPredicate = RDFResource(Relation.semanticSimilarity.name, False, self.__namespaces[lac])
            nestedObject = RDFResource(nid2, False, self.__namespaces[lac])

            subject1 = Triplet(nestedSubject, nestedPredicate, nestedObject)
            subject2 = Triplet(nestedObject, nestedPredicate, nestedSubject)

            predicate = RDFResource(Relation.certainty.name, False, self.__namespaces[lac])
            objct = RDFResource(score)

            self.set_neighbors(Relation.semanticSimilarity.name, nid1, nid2, score)
            self.set_neighbors(Relation.semanticSimilarity.name, nid2, nid1, score)

            triplet1 = Triplet(subject1, predicate, objct)
            triplet2 = Triplet(subject2, predicate, objct)

            self.__triplets.extend([subject1, subject2, triplet1, triplet2])
            #self.__triplets.extend([subject1, triplet1])  # print instead

        print('** ** Creating the semantic similarity relations has started ** ** ')
        self.__triplets = []  # to remove
        similar_pairs = self.word_embeddings.get_affinity_between_column_names(self.__id_to_labels)
        for col1_id, col2_id, distance in similar_pairs:
            if distance >= 0.5:
                create_triplets(col1_id, col2_id, distance)
        self.__triplets_to_dump.extend(self.__triplets)

    def build_content_sim_mh_text(self, mh_signatures):

        def create_triplets(nid1, nid2, score):
            nestedSubject = RDFResource(nid1, False, self.__namespaces[lac])
            nestedPredicate = RDFResource(Relation.contentSimilarity.name, False, self.__namespaces[lac])
            nestedObject = RDFResource(nid2, False, self.__namespaces[lac])

            subject1 = Triplet(nestedSubject, nestedPredicate, nestedObject)
            subject2 = Triplet(nestedObject, nestedPredicate, nestedSubject)

            predicate = RDFResource(Relation.certainty.name, False, self.__namespaces[lac])
            objct = RDFResource(score)

            triplet1 = Triplet(subject1, predicate, objct)
            triplet2 = Triplet(subject2, predicate, objct)

            self.set_neighbors(Relation.contentSimilarity.name, nid1, nid2, score)
            self.set_neighbors(Relation.contentSimilarity.name, nid2, nid1, score)

            self.__triplets.extend([subject1, subject2, triplet1, triplet2])
            #self.__triplets.extend([subject1, triplet1])

        print('** ** Creating the text content similarity relations has started ** ** ')
        self.__triplets = []
        # Materialize signatures for convenience
        mh_sig_obj = []

        content_index = MinHashLSH(threshold=0.7, num_perm=512)

        # Create minhash objects and index
        for nid, mh_sig in mh_signatures:
            mh_obj = MinHash(num_perm=512)
            #print(type(mh_sig), len(mh_sig))
            mh_array = np.asarray(mh_sig, dtype=int)
            mh_obj.hashvalues = mh_array
            content_index.insert(nid, mh_obj)
            mh_sig_obj.append((nid, mh_obj))

        # Query objects
        for nid, mh_obj in mh_sig_obj:
            res = content_index.query(mh_obj)
            for r_nid in res:
                if r_nid != nid:
                    r_mh_obj = list(filter(lambda x: x[0] == r_nid, mh_sig_obj))[0][1]
                    distance = mh_obj.jaccard(r_mh_obj)
                    create_triplets(nid, r_nid, distance)
        self.__triplets_to_dump.extend(self.__triplets)
        print('** ** Creating the text content similarity relations has ended ** ** ')
        return content_index

    def build_content_sim_relation_num_overlap_distr(self, id_sig):

        def compute_overlap(ref_left, ref_right, left, right):
            ov = 0
            if left >= ref_left and right <= ref_right:
                ov = float((right - left) / (ref_right - ref_left))
            elif left >= ref_left and left <= ref_right:
                domain_ov = ref_right - left
                ov = float(domain_ov / (ref_right - ref_left))
            elif right <= ref_right and right >= ref_left:
                domain_ov = right - ref_left
                ov = float(domain_ov / (ref_right - ref_left))
            return float(ov)

        def create_triplets(nid1, nid2, score, inddep=False):
            nestedSubject = RDFResource(nid1, False, self.__namespaces[lac])
            nestedObject = RDFResource(nid2, False, self.__namespaces[lac])

            if inddep is False:
                nestedPredicate = RDFResource(Relation.contentSimilarity.name, False, self.__namespaces[lac])
                self.set_neighbors(Relation.contentSimilarity.name, nid1, nid2, score)
                self.set_neighbors(Relation.contentSimilarity.name, nid2, nid1, score)
            else:
                nestedPredicate = RDFResource(Relation.inclusionDependency.name, False, self.__namespaces[lac])
                self.set_neighbors(Relation.inclusionDependency.name, nid1, nid2, score)
                self.set_neighbors(Relation.inclusionDependency.name, nid2, nid1, score)

            subject1 = Triplet(nestedSubject, nestedPredicate, nestedObject)
            subject2 = Triplet(nestedObject, nestedPredicate, nestedSubject)

            predicate = RDFResource(Relation.certainty.name, False, self.__namespaces[lac])
            objct = RDFResource(score)

            triplet1 = Triplet(subject1, predicate, objct)
            triplet2 = Triplet(subject2, predicate, objct)

            self.__triplets.extend([subject1, subject2, triplet1, triplet2])
            #self.__triplets.extend([subject1, triplet1])  # for metadata

        def get_info_for(nids):
            info = []
            for nid in nids:
                db_name, source_name, field_name, data_type = self._column_id_to_name[nid]
                info.append((nid, db_name, source_name, field_name))
            return info

        print('** ** Creating the numerical content similarity relations has started ** **')
        self.__triplets = []
        overlap = 0.95

        fields = []
        domains = []
        stats = []
        for c_k, (c_median, c_iqr, c_min_v, c_max_v) in id_sig:
            fields.append(c_k)
            domain = (c_median + c_iqr) - (c_median - c_iqr)
            domains.append(domain)
            extreme_left = c_median - c_iqr
            min = c_min_v
            extreme_right = c_median + c_iqr
            max = c_max_v
            stats.append((min, extreme_left, extreme_right, max))

        zipped_and_sorted = sorted(zip(domains, fields, stats), reverse=True)
        candidate_entries = [(y, x, z[0], z[1], z[2], z[3]) for (x, y, z) in zipped_and_sorted]
        single_points = []
        for ref in candidate_entries:
            ref_nid, ref_domain, ref_x_min, ref_x_left, ref_x_right, ref_x_max = ref

            if ref_nid == '2314808454':
                debug = True

            if ref_domain == 0:
                single_points.append(ref)

            info1 = get_info_for([ref_nid])

            (nid, db_name, source_name, field_name) = info1[0]
            for entry in candidate_entries:
                candidate_nid, candidate_domain, candidate_x_min, candidate_x_left, candidate_x_right, candidate_x_max = entry
                if candidate_nid == '1504465753':
                    debug = True

                if candidate_nid == ref_nid:
                    continue

                if ref_domain == 0:
                    continue
                # Check for filtered inclusion dependencies first
                if isinstance(candidate_domain, float) or isinstance(candidate_domain, int):  # Filter these out
                    # Check ind. dep.
                    info2 = get_info_for([candidate_nid])
                    (_, _, sn1, fn1) = info1[0]
                    (_, _, sn2, fn2) = info2[0]
                    if isinf(float(ref_x_min)) or isinf(float(ref_x_max)) or isinf(float(candidate_x_max)) or isinf(
                            float(candidate_x_min)):
                        continue
                    if candidate_x_min >= ref_x_min and candidate_x_max <= ref_x_max:
                        # inclusion relation
                            actual_overlap = compute_overlap(ref_x_left, ref_x_right, candidate_x_left,
                                                             candidate_x_right)

                            if actual_overlap >= 0.9:
                                create_triplets(candidate_nid, ref_nid, actual_overlap, inddep=True)

                # if float(candidate_domain / ref_domain) <= overlap:
                #    # There won't be a content sim relation -> not even the entire domain would overlap more than the th.
                #    break
                actual_overlap = compute_overlap(ref_x_left, ref_x_right, candidate_x_left, candidate_x_right)
                if actual_overlap >= overlap:
                    create_triplets(candidate_nid, ref_nid, actual_overlap)

        # Final clustering for single points
        fields = []
        medians = []

        for (nid, domain, x_min, x_left, x_right, x_max) in single_points:
            median = x_right - float(x_right / 2)
            fields.append(nid)
            medians.append(median)

        x_median = np.asarray(medians)
        x_median = x_median.reshape(-1, 1)

        # At this point, we may have not found any points at all, in which case we can
        # safely exit
        if len(x_median) == 0:
            self.__triplets_to_dump.extend(self.__triplets)
            print('** ** Creating the numerical content similarity relations has ended ** ** ')
            return

        db_median = DBSCAN(eps=0.1, min_samples=2).fit(x_median)
        labels_median = db_median.labels_
        n_clusters = len(set(labels_median)) - (1 if -1 in labels_median else 0)
        # print("#clusters: " + str(n_clusters))

        clusters_median = defaultdict(list)
        for i in range(len(labels_median)):
            clusters_median[labels_median[i]].append(i)

        for k, v in clusters_median.items():
            if k == -1:
                continue
            # print("Cluster: " + str(k))
            for el in v:
                nid = fields[el]
                info = get_info_for([nid])
                (nid, db_name, source_name, field_name) = info[0]
                # print(source_name + " - " + field_name + " median: " + str(medians[el]))
                for el2 in v:
                    if el != el2:
                        nid1 = fields[el]
                        nid2 = fields[el2]
                        create_triplets(nid1, nid2, overlap)
        self.__triplets_to_dump.extend(self.__triplets)
        print('** ** Creating the numerical content similarity relations has ended ** ** ')

    def build_pkfk_relation(self):

        def get_data_type_of(nid):
            _, _, _, data_type = self._column_id_to_name[nid]
            return data_type

        def get_neighborhood(n):
            neighbors = []
            data_type = get_data_type_of(n)
            if data_type == "N":
                neighbors = self.neighbors_id(n, Relation.inclusionDependency)
            if data_type == "T":
                neighbors = self.neighbors_id(n, Relation.contentSimilarity)
            return neighbors

        def iterate_ids():
            for k, _ in self._column_id_to_name.items():
                yield k

        def create_triplets(nid1, nid2, score):
            nestedSubject = RDFResource(nid1, False, self.__namespaces[lac])
            nestedPredicate = RDFResource(Relation.pkfk.name, False, self.__namespaces[lac])
            nestedObject = RDFResource(nid2, False, self.__namespaces[lac])

            subject1 = Triplet(nestedSubject, nestedPredicate, nestedObject)
            subject2 = Triplet(nestedObject, nestedPredicate, nestedSubject)

            predicate = RDFResource(Relation.certainty.name, False, self.__namespaces[lac])
            objct = RDFResource(score)

            triplet1 = Triplet(subject1, predicate, objct)
            triplet2 = Triplet(subject2, predicate, objct)
            
            self.__triplets.extend([subject1, subject2, triplet1, triplet2]) 
            #self.__triplets.extend([subject1, triplet1])  # for the associated metadata

        print('** ** Creating the primary keys / foreign keys relations has started ** **')
        self.__triplets = []
        total_pkfk_relations = 0
        for n in iterate_ids():
            if not 'cardinality' in self.__neighbors[n]:
                continue
            n_card = self.__neighbors[n]['cardinality']
            if n == '2314808454' or n == '1504465753':
                debug = True
            if n_card > 0.7:  # Early check if this is a candidate
                neighborhood = get_neighborhood(n)
                for ne in neighborhood:
                    if (n == '1280022251' and ne[0] == '458928166'):
                        print(n_card, 'pkfk----------------------------')
                    if ne[0] is not n:
                        if ne[0] == '1504465753' or ne[0] == '2314808454':
                            debug = True
                        ne_card = self.__neighbors[ne[0]]['cardinality']
                        if n_card > ne_card:
                            highest_card = n_card
                        else:
                            highest_card = ne_card
                        # if ne_card < 0.5:
                        create_triplets(n, ne[0], highest_card)
                        total_pkfk_relations += 1
        self.__triplets_to_dump.extend(self.__triplets)
        print('** ** Creating the primary keys / foreign keys relations has ended ** **')
