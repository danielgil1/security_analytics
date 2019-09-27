RANDOM_STATE            =   42

TRAIN_DATA_FLOW         =   "../inputs/a2_feat_flow_level_normal.csv"
TEST_DATA_FLOW          =   "../inputs/a2_feat_flow_level_test.csv"
TRAIN_DATA_PACKET       =   "../inputs/a2_feat_packet_level_normal.csv"
TEST_DATA_PACKET        =   "../inputs/a2_feat_packet_level_test.csv"

SELECTED_FEATURES_FLOW  =   ['total_duration','total_bytes', 'total_packets', 'src_ports', 'dst_ports', 'pps', 'bps','bpp']
SELECTED_FEATURES_PACKET=   ['total_bytes', 'total_packets']