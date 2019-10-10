import pandas as pd
import pickle
import itertools
import cconfig

def generate_basic_features(dataset):
    colFlows=list()
    i=0
    for index,data in dataset.groupby(['tcp_stream','protocol']):
        i+=1
        
        flow_list=list()
        # go through conversations (unidirectional flows) to generate the flows
        for index_conversation,conversation in data.iterrows():
            # check if conversation already exists
            isnewflow=True
            
            for f in flow_list:
                if (conversation.src_ip==f.get('src_ip',"") and conversation.src_port==f.get('src_port',"")) and (conversation.dst_ip==f.get('dst_ip',"") and conversation.dst_port==f.get('dst_port',"")):
                    isnewflow=False

                    # set up basic features
                    f['fwd_packets']=f.get('fwd_packets',0)+conversation.total_packets
                    f['fwd_bytes']=f.get('fwd_bytes',0)+conversation.total_bytes
                    f['fwd_duration']=f.get('fwd_duration',0)+conversation.total_duration

                    #TODO: set timestamps
                    f['flow_start']=conversation.flow_start

                    # set up rate-based features
                    f['fwd_bps']=f.get('fwd_bps',0)+conversation.bps
                    f['fwd_pps']=f.get('fwd_pps',0)+conversation.pps
                    f['fwd_bpp']=f.get('fwd_bpp',0)+conversation.bpp

                    # set up error-based features 
                    f['fwd_http_errors']=f.get('fwd_http_errors',0)+conversation.total_http_errors
                    f['fwd_failures']=f.get('fwd_failures',0)+conversation.total_failures

                    # set up tcp_flags-based features
                    f['fwd_flag_syn']=f.get('fwd_flag_syn',0)+conversation.flag_syn
                    f['fwd_flag_ack']=f.get('fwd_flag_ack',0)+conversation.flag_ack
                    f['fwd_flag_fin']=f.get('fwd_flag_fin',0)+conversation.flag_fin
                    f['fwd_flag_psh']=f.get('fwd_flag_psh',0)+conversation.flag_psh
                    f['fwd_flag_rst']=f.get('fwd_flag_rst',0)+conversation.flag_rst
                    f['first_flag']=conversation.first_flag

                    # statistical features from packets
                    f['fwd_avg_bytes']=conversation.avg_bytes
                    f['fwd_stdev_bytes']=conversation.stdev_bytes
                    f['fwd_min_bytes']=conversation.min_bytes
                    f['fwd_max_bytes']=conversation.max_bytes


                elif (conversation.src_ip==f.get('dst_ip',"") and conversation.src_port==f.get('dst_port',"")) and (conversation.dst_ip==f.get('src_ip',"") and conversation.dst_port==f.get('src_port',"")):
                    isnewflow=False
                    
                    # set up basic features
                    f['bwd_packets']=f.get('bwd_packets',0)+conversation.total_packets
                    f['bwd_bytes']=f.get('bwd_bytes',0)+conversation.total_bytes
                    f['bwd_duration']=f.get('bwd_duration',0)+conversation.total_duration

                    #TODO: set timestamps
                    f['flow_finish']=conversation.flow_finish

                    # set up rate-based features
                    f['bwd_bps']=f.get('bwd_bps',0)+conversation.bps
                    f['bwd_pps']=f.get('bwd_pps',0)+conversation.pps
                    f['bwd_bpp']=f.get('bwd_bpp',0)+conversation.bpp
                    
                    # set up error-based features
                    f['fwd_http_errors']=f.get('fwd_http_errors',0)+conversation.total_http_errors
                    f['fwd_failures']=f.get('fwd_failures',0)+conversation.total_failures

                    # set up tcp_flags-based features
                    f['bwd_flag_syn']=f.get('bwd_flag_syn',0)+conversation.flag_syn
                    f['bwd_flag_ack']=f.get('bwd_flag_ack',0)+conversation.flag_ack
                    f['bwd_flag_fin']=f.get('bwd_flag_fin',0)+conversation.flag_fin
                    f['bwd_flag_psh']=f.get('bwd_flag_psh',0)+conversation.flag_psh
                    f['bwd_flag_rst']=f.get('bwd_flag_rst',0)+conversation.flag_rst
                    f['last_flag']=conversation.last_flag

                    # statistical features from packets
                    f['bwd_avg_bytes']=conversation.avg_bytes
                    f['bwd_stdev_bytes']=conversation.stdev_bytes
                    f['bwd_min_bytes']=conversation.min_bytes
                    f['bwd_max_bytes']=conversation.max_bytes

            if isnewflow:
                new_flow={}
                
                # set up basic features for new flow
                new_flow['src_ip']=conversation.src_ip
                new_flow['dst_ip']=conversation.dst_ip
                new_flow['src_port']=conversation.src_port
                new_flow['dst_port']=conversation.dst_port
                new_flow['fwd_packets']=conversation.total_packets
                new_flow['bwd_packets']=0
                new_flow['fwd_bytes']=conversation.total_bytes
                new_flow['bwd_bytes']=0
                new_flow['fwd_duration']=new_flow.get('fwd_duration',0)+conversation.total_duration
                
                

                # set timestamps
                new_flow['flow_start']=conversation.flow_start

                # set up rate-based features for a new flow
                new_flow['fwd_bps']=conversation.bps
                new_flow['fwd_pps']=conversation.pps
                new_flow['fwd_bpp']=conversation.bpp

                # set up error-based features
                new_flow['fwd_total_http_errors']=conversation.total_http_errors
                new_flow['fwd_total_failures']=conversation.total_failures
                new_flow['tcp_stream']=index[0]
                new_flow['protocol']=index[1]

                # set up tcp_flags-based features
                new_flow['fwd_flag_syn']=conversation.flag_syn
                new_flow['fwd_flag_ack']=conversation.flag_ack
                new_flow['fwd_flag_fin']=conversation.flag_fin
                new_flow['fwd_flag_psh']=conversation.flag_psh
                new_flow['fwd_flag_rst']=conversation.flag_rst
                new_flow['first_flag']=conversation.first_flag
                
                
                # statistical features from packets
                new_flow['fwd_avg_bytes']=conversation.avg_bytes
                new_flow['fwd_stdev_bytes']=conversation.stdev_bytes
                new_flow['fwd_min_bytes']=conversation.min_bytes
                new_flow['fwd_max_bytes']=conversation.max_bytes
                
                flow_list.append(new_flow)
            
            
        colFlows.append(flow_list)
        
        
    return colFlows


# Time based features

def time_get_count_dest(timebase,src_ip,data_flow):
    delta=pd.Timedelta(cconfig.LAST_N_SECONDS)
    query_time=timebase-delta
    count_dest=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==src_ip)].dst_ip.unique())
    
    return count_dest

def time_get_count_src(timebase,dst_ip,data_flow):
    delta=pd.Timedelta(cconfig.LAST_N_SECONDS)
    query_time=timebase-delta
    count_src=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.dst_ip==dst_ip)].src_ip.unique())
    
    return count_src

def time_get_count_serv_src(timebase,src_ip,dst_port,data_flow):
    delta=pd.Timedelta(cconfig.LAST_N_SECONDS)
    query_time=timebase-delta
    count_serv_src=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==src_ip) & (data_flow.dst_port==dst_port)])
    
    return count_serv_src

def time_get_count_serv_dst(timebase,dst_ip,src_port,data_flow):
    delta=pd.Timedelta(cconfig.LAST_N_SECONDS)
    query_time=timebase-delta
    count_serv_dst=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==dst_ip) & (data_flow.dst_port==src_port)])
    
    return count_serv_dst

###################

# Connection-based features
def get_count_dest_conn(timebase,src_ip,data_flow):
    """ Number of flows to unique destination IPs in the last N flows
        from the same source
    """
    n_flows=cconfig.LAST_N_FLOWS
    count_dest_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.src_ip==src_ip)].tail(n_flows).dst_ip.unique())
    
    return count_dest_conn

def get_count_src_conn(timebase,dst_ip,data_flow):
    """ Number of flows from unique source IPs in the last N flows to the same destination
    """
    n_flows=cconfig.LAST_N_FLOWS
    count_src_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.dst_ip==dst_ip)].tail(n_flows).src_ip.unique())
    
    return count_src_conn


def get_count_serv_src_conn(timebase,src_ip,dst_port,data_flow):
    """ Number of flows from the source IP to the same destination port in the last N flows
    """
    n_flows=cconfig.LAST_N_FLOWS
    count_serv_src_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.src_ip==src_ip) & (data_flow.dst_port==dst_port)].tail(n_flows))
    
    return count_serv_src_conn


def get_count_serv_dst_conn(timebase,dst_ip,src_port,data_flow):
    """ Number of flows to the destination IP to the same source port in the last N flows
    """
    n_flows=cconfig.LAST_N_FLOWS
    count_serv_dst_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.src_ip==dst_ip) & (data_flow.dst_port==src_port)].tail(n_flows))
    
    return count_serv_dst_conn


if __name__ == "__main__":
    pd.set_option("display.precision", 50)

    print("Loading training set.")
    df_normal=pd.read_csv("../inputs/training.csv")
    df_normal=df_normal[:1000]

    df_normal=df_normal.sort_values(by=['tcp_stream','protocol','start_time'])
    df_normal[['flow_start']]=df_normal[['flow_start']].apply(pd.to_datetime)
    df_normal[['flow_finish']]=df_normal[['flow_finish']].apply(pd.to_datetime)
    
    print("Generating basic features.")
    flows_normal=generate_basic_features(df_normal)
    merged_normal = list(itertools.chain.from_iterable(flows_normal))
    df_merged_normal=pd.DataFrame.from_dict(merged_normal)
    
    print("Generating time features.")
    df_merged_normal[['flow_start']]=df_merged_normal[['flow_start']].apply(pd.to_datetime)
    df_merged_normal[['flow_finish']]=df_merged_normal[['flow_finish']].apply(pd.to_datetime)
    df_merged_normal['flow_duration']=(df_merged_normal.flow_finish-df_merged_normal.flow_start).dt.total_seconds() 
    df_merged_normal['count_dest']=df_merged_normal.apply(lambda x: time_get_count_dest(x.flow_start,x.src_ip,df_merged_normal),axis=1)
    df_merged_normal['count_src']=df_merged_normal.apply(lambda x: time_get_count_src(x.flow_start,x.dst_ip,df_merged_normal),axis=1)
    df_merged_normal['count_serv_src']=df_merged_normal.apply(lambda x: time_get_count_serv_src(x.flow_start,x.src_ip,x.dst_port,df_merged_normal),axis=1)
    df_merged_normal['count_serv_dst']=df_merged_normal.apply(lambda x: time_get_count_serv_dst(x.flow_start,x.dst_ip,x.src_port,df_merged_normal),axis=1)
    
    print("Generating connection features")
    df_merged_normal['count_dest_conn']=df_merged_normal.apply(lambda x: get_count_dest_conn(x.flow_start,x.src_ip,test),axis=1)
    df_merged_normal['count_src_conn']=df_merged_normal.apply(lambda x: get_count_src_conn(x.flow_start,x.dst_ip,test),axis=1)
    df_merged_normal['count_serv_src_conn']=df_merged_normal.apply(lambda x: get_count_serv_src_conn(x.flow_start,x.src_ip,x.dst_port,test),axis=1)
    df_merged_normal['count_count_serv_src_conn']=df_merged_normal.apply(lambda x: get_count_serv_src_conn(x.flow_start,x.dst_ip,x.src_port,test),axis=1)

    print("Saving training dataset into dataframe pickle.")
    with open('df_flows_normal.pickle', 'wb') as handle:
        pickle.dump(df_merged_normal, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Loading training set.")
    df_attack=pd.read_csv("../inputs/testing.csv")
    df_attack=df_attack[:1000]

    df_attack=df_attack.sort_values(by=['tcp_stream','protocol','start_time'])
    df_attack[['flow_start']]=df_attack[['flow_start']].apply(pd.to_datetime)
    df_attack[['flow_finish']]=df_attack[['flow_finish']].apply(pd.to_datetime)
    
    print("Generating basic features.")
    flows_attack=generate_basic_features(df_attack)
    merged_attack = list(itertools.chain.from_iterable(flows_attack))
    df_merged_attack=pd.DataFrame.from_dict(merged_attack)

    print("Generating time features.")
    df_merged_attack[['flow_start']]=df_merged_attack[['flow_start']].apply(pd.to_datetime)
    df_merged_attack[['flow_finish']]=df_merged_attack[['flow_finish']].apply(pd.to_datetime)
    df_merged_attack['flow_duration']=(df_merged_attack.flow_finish-df_merged_attack.flow_start).dt.total_seconds() 
    df_merged_attack['count_dest']=df_merged_attack.apply(lambda x: time_get_count_dest(x.flow_start,x.src_ip,df_merged_attack),axis=1)
    df_merged_attack['count_src']=df_merged_attack.apply(lambda x: time_get_count_src(x.flow_start,x.dst_ip,df_merged_attack),axis=1)
    df_merged_attack['count_serv_src']=df_merged_attack.apply(lambda x: time_get_count_serv_src(x.flow_start,x.src_ip,x.dst_port,df_merged_attack),axis=1)
    df_merged_attack['count_serv_dst']=df_merged_attack.apply(lambda x: time_get_count_serv_dst(x.flow_start,x.dst_ip,x.src_port,df_merged_attack),axis=1)
    
    print("Generating connection features")
    df_merged_attack['count_dest_conn']=df_merged_attack.apply(lambda x: get_count_dest_conn(x.flow_start,x.src_ip,test),axis=1)
    df_merged_attack['count_src_conn']=df_merged_attack.apply(lambda x: get_count_src_conn(x.flow_start,x.dst_ip,test),axis=1)
    df_merged_attack['count_serv_src_conn']=df_merged_attack.apply(lambda x: get_count_serv_src_conn(x.flow_start,x.src_ip,x.dst_port,test),axis=1)
    df_merged_attack['count_count_serv_src_conn']=df_merged_attack.apply(lambda x: get_count_serv_src_conn(x.flow_start,x.dst_ip,x.src_port,test),axis=1)

    print("Saving tesing dataset into dataframe pickle.")
    with open('df_flows_attack.pickle', 'wb') as handle:
            pickle.dump(df_merged_attack, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
