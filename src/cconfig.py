RANDOM_STATE            =   42

TRAIN_DATA_FLOW         =   "../inputs/A2_feat_flow_level_normal_v2.csv"
TEST_DATA_FLOW          =   "../inputs/A2_feat_flow_level_test_v2.csv"
TRAIN_DATA_PACKET       =   "../inputs/a2_feat_packet_level_normal.csv"
TEST_DATA_PACKET        =   "../inputs/a2_feat_packet_level_test.csv"

SELECTED_FEATURES_FLOW  =   ['total_duration','total_bytes', 'total_packets', 'src_ports', 'dst_ports', 'pps', 'bps','bpp','num_protocols', 'flag_ack', 'flag_fin', 'flag_psh', 'flag_rst',
       'flag_syn', 'attack']
SELECTED_FEATURES_PACKET=   ['total_bytes', 'total_packets']