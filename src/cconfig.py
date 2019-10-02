RANDOM_STATE            =   42
DEFAULT_NUM_CLUSTERS    =   5

DATASET_TYPE_FLOW       =   "FLOW"
DATASET_TYPE_PACKET     =   "PACKET"
TRAIN_DATA_FLOW         =   "../inputs/A2_feat_flow_level_normal_v2.csv"
TEST_DATA_FLOW          =   "../inputs/A2_feat_flow_level_test_v2.csv"
TRAIN_DATA_PACKET       =   "../inputs/a2_feat_packet_level_normal_2.csv"
TEST_DATA_PACKET        =   "../inputs/a2_feat_packet_level_attack_2.csv"

SELECTED_FEATURES_FLOW  =   ['total_duration','total_bytes', 'total_packets', 'src_ports', 'dst_ports', 'pps', 'bps','bpp','num_protocols', 'flag_ack', 'flag_fin', 'flag_psh', 'flag_rst',
       'flag_syn']

SELECTED_FEATURES_PACKET=   ['duration', 'dc_src_ports',
       'dc_dst_ports', 'total_mb', 'avg_mb', 'mbps', 'total_packets',
       'num_protocols', 'flag_ack', 'flag_fin', 'flag_psh', 'flag_rst',
       'flag_syn']