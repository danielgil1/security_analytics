RANDOM_STATE            =   42
DEFAULT_NUM_CLUSTERS    =   5

LAST_N_FLOWS            =   100
LAST_N_SECONDS          =   "30 seconds"

DATASET_TYPE_FLOW       =   "FLOW"
DATASET_TYPE_PACKET     =   "PACKET"
DATASET_TYPE_BIFLOW     =   "BIFLOW"

TRAIN_DATA_FLOW         =   "../inputs/A2_feat_flow_level_normal_v2.csv"
TEST_DATA_FLOW          =   "../inputs/A2_feat_flow_level_test_v2.csv"
TRAIN_DATA_PACKET       =   "../inputs/a2_feat_packet_level_normal_3.csv"
TEST_DATA_PACKET        =   "../inputs/a2_feat_packet_level_attack_3.csv"
TRAIN_DATA_BIFLOW       =   "../inputs/df_flows_normal.pickle"
TEST_DATA_BIFLOW        =   "../inputs/df_flows_attack.pickle"

SELECTED_FEATURES_BIFLOW=   ['fwd_packets',
       'bwd_packets', 'fwd_bytes', 'bwd_bytes', 'fwd_duration', 
       'fwd_bps', 'fwd_pps', 'fwd_bpp', 'fwd_total_http_errors',
       'fwd_total_failures', 'fwd_flag_syn',
       'fwd_flag_ack', 'fwd_flag_fin', 'fwd_flag_psh', 'fwd_flag_rst',
       'fwd_avg_bytes', 'fwd_stdev_bytes', 'fwd_min_bytes',
       'fwd_max_bytes', 'bwd_duration','bwd_bps', 'bwd_pps',
       'bwd_bpp', 'fwd_http_errors', 'fwd_failures', 'bwd_flag_syn',
       'bwd_flag_ack', 'bwd_flag_fin', 'bwd_flag_psh', 'bwd_flag_rst',
        'bwd_avg_bytes', 'bwd_stdev_bytes', 'bwd_min_bytes',
       'bwd_max_bytes', 'flow_duration', 'count_dest', 'count_src',
       'count_serv_src', 'count_serv_dst', 'count_dest_conn', 'count_src_conn',
       'count_serv_src_conn', 'count_count_serv_src_conn']

SELECTED_FEATURES_BIFNH= ['bwd_packets', 'fwd_bytes', 'bwd_bytes', 'fwd_duration', 
       'fwd_bps', 'fwd_pps', 'fwd_bpp', 'fwd_flag_syn',
       'fwd_flag_ack', 'fwd_flag_fin', 'fwd_flag_psh', 'fwd_flag_rst',
       'fwd_avg_bytes', 'fwd_stdev_bytes', 'fwd_min_bytes',
       'fwd_max_bytes', 'bwd_duration', 'bwd_bps', 'bwd_pps',
       'bwd_bpp', 'bwd_flag_syn',
       'bwd_flag_ack', 'bwd_flag_fin', 'bwd_flag_psh', 'bwd_flag_rst',
        'bwd_avg_bytes', 'bwd_stdev_bytes', 'bwd_min_bytes',
       'bwd_max_bytes', 'flow_duration', 'count_dest', 'count_src',
       'count_serv_src', 'count_serv_dst', 'count_dest_conn', 'count_src_conn',
       'count_serv_src_conn', 'count_count_serv_src_conn']

SELECTED_FEATURES_BIFNF=['bwd_packets', 'fwd_bytes', 'bwd_bytes', 'fwd_duration', 
       'fwd_bps', 'fwd_pps', 'fwd_bpp', 
       'fwd_avg_bytes', 'fwd_stdev_bytes', 'fwd_min_bytes',
       'fwd_max_bytes', 'bwd_duration', 'bwd_bps', 'bwd_pps',
       'bwd_bpp',
        'bwd_avg_bytes', 'bwd_stdev_bytes', 'bwd_min_bytes',
       'bwd_max_bytes', 'flow_duration', 'count_dest', 'count_src',
       'count_serv_src', 'count_serv_dst', 'count_dest_conn', 'count_src_conn',
       'count_serv_src_conn', 'count_count_serv_src_conn'] 

SELECTED_FEATURES_PACKET=   ['total_duration','total_bytes','total_packets','src_ports','dst_ports','pps','bps','bpp','num_protocols','total_http_errors','total_failures','flag_syn','flag_ack']

FLOW_ANOMALIES_SORT     =   ['total_bytes','total_packets','total_duration']
BIFLOW_ANOMALIES_SORT   =   ['fwd_packets','bwd_packets', 'fwd_bytes', 'bwd_bytes', 'fwd_duration']
