import os.path
import time
from helpers.helper import time_taken, upload_glac, drop_glac
from rdf_builder import RDFBuilder
from storage.elasticsearch_client import ElasticsearchClient


def main(output_path=None):
    start_all = time.time()
    rdfBuilder = RDFBuilder()
    store = ElasticsearchClient()

    # Get all fields from store
    fields_gen = store.get_profile_attributes()

    # Network skeleton and hierarchical relations (table - field), etc
    start_schema = time.time()
    rdfBuilder.initialize_nodes(fields_gen)
    end_schema = time.time()
    print("Total skeleton: {0}".format(str(end_schema - start_schema)))
    print("!!1 " + str(end_schema - start_schema))

    # Schema_sim relation
    start_schema_sim = time.time()
    schema_sim_index = rdfBuilder.build_semantic_sim_relation()
    end_schema_sim = time.time()
    print("Total schema-sim: {0}".format(str(end_schema_sim - start_schema_sim)))
    print("!!2 " + str(end_schema_sim - start_schema_sim))

    # Content_sim text relation (minhash-based)
    start_text_sig_sim = time.time()
    st = time.time()
    mh_signatures = store.get_profiles_minhash()
    et = time.time()
    print("Time to extract minhash signatures from store: {0}".format(str(et - st)))
    print("!!3 " + str(et - st))

    content_sim_index = rdfBuilder.build_content_sim_mh_text(mh_signatures)
    end_text_sig_sim = time.time()
    print("Total text-sig-sim (minhash): {0}".format(str(end_text_sig_sim - start_text_sig_sim)))
    print("!!4 " + str(end_text_sig_sim - start_text_sig_sim))

    # Content_sim num relation
    start_num_sig_sim = time.time()
    id_sig = store.get_num_stats()
    # networkbuilder.build_content_sim_relation_num(network, id_sig)
    rdfBuilder.build_content_sim_relation_num_overlap_distr(id_sig)
    # networkbuilder.build_content_sim_relation_num_overlap_distr_indexed(network, id_sig)
    end_num_sig_sim = time.time()
    print("Total num-sig-sim: {0}".format(str(end_num_sig_sim - start_num_sig_sim)))
    print("!!5 " + str(end_num_sig_sim - start_num_sig_sim))

    # Primary Key / Foreign key relation
    start_pkfk = time.time()
    rdfBuilder.build_pkfk_relation()
    end_pkfk = time.time()
    print("Total PKFK: {0}".format(str(end_pkfk - start_pkfk)))
    print("!!6 " + str(end_pkfk - start_pkfk))

    end_all = time.time()
    print("Total time: {0}".format(str(end_all - start_all)))
    print("!!7 " + str(end_all - start_all))

    rdfBuilder.dump_into_file(output_path + 'glac.ttls')
    print("Graph generated in ", time_taken(start_all, end_all))


if __name__ == "__main__":
    path = 'out/'
    if os.path.exists('out/glac.ttls'):
        os.remove('out/glac.ttls')
    main(path)
    drop_glac(namespace='glac')
    upload_glac('out/glac.ttls', namespace='glac')
