import pandas as pd
import pickle
import itertools

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

                elif (conversation.src_ip==f.get('dst_ip',"") and conversation.src_port==f.get('dst_port',"")) and (conversation.dst_ip==f.get('src_ip',"") and conversation.dst_port==f.get('src_port',"")):
                    isnewflow=False
                    
                    # set up basic features
                    f['bwd_packets']=f.get('bwd_packets',0)+conversation.total_packets
                    f['bwd_bytes']=f.get('bwd_bytes',0)+conversation.total_bytes
                    f['bwd_duration']=f.get('bwd_duration',0)+conversation.total_duration

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
                
                # set up rate-based features for a new flow
                new_flow['fwd_bps']=new_flow.get('fwd_bps',0)+conversation.bps
                new_flow['fwd_pps']=new_flow.get('fwd_pps',0)+conversation.pps
                new_flow['fwd_bpp']=new_flow.get('fwd_bpp',0)+conversation.bpp

                # set up error-based features
                new_flow['fwd_total_http_errors']=conversation.total_http_errors
                new_flow['fwd_total_failures']=conversation.total_failures
                new_flow['tcp_stream']=index[0]
                new_flow['protocol']=index[1]

                # set up tcp_flags-based features
                f['fwd_flag_syn']=f.get('fwd_flag_syn',0)+conversation.flag_syn
                f['fwd_flag_ack']=f.get('fwd_flag_ack',0)+conversation.flag_ack
                f['fwd_flag_fin']=f.get('fwd_flag_fin',0)+conversation.flag_fin
                f['fwd_flag_psh']=f.get('fwd_flag_psh',0)+conversation.flag_psh
                f['fwd_flag_rst']=f.get('fwd_flag_rst',0)+conversation.flag_rst

                flow_list.append(new_flow)
            
            
        colFlows.append(flow_list)
        
        
    return colFlows


# Time based features

def time_get_count_dest(timebase,src_ip,data_flow):
    delta=pd.Timedelta('5 seconds')
    query_time=timebase-delta
    count_dest=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==src_ip)].dst_ip.unique())
    
    return count_dest

def time_get_count_src(timebase,dst_ip,data_flow):
    delta=pd.Timedelta('5 seconds')
    query_time=timebase-delta
    count_src=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.dst_ip==dst_ip)].src_ip.unique())
    
    return count_src

def time_get_count_serv_src(timebase,src_ip,dst_port,data_flow):
    delta=pd.Timedelta('5 seconds')
    query_time=timebase-delta
    count_serv_src=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==src_ip) & (data_flow.dst_port==dst_port)])
    
    return count_serv_src

def time_get_count_serv_dst(timebase,dst_ip,src_port,data_flow):
    delta=pd.Timedelta('5 seconds')
    query_time=timebase-delta
    count_serv_dst=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==dst_ip) & (data_flow.dst_port==src_port)])
    
    return count_serv_dst

###################


if __name__ == "__main__":
    pd.set_option("display.precision", 50)

    df_normal=pd.read_csv("../inputs/training.csv")
    df_normal=df_normal.sort_values(by=['tcp_stream','protocol','start_time'])
    df_normal[['flow_start']]=df_normal[['flow_start']].apply(pd.to_datetime)
    df_normal[['flow_finish']]=df_normal[['flow_finish']].apply(pd.to_datetime)
    flows_normal=generate_basic_features(df_normal)




    merged_normal = list(itertools.chain.from_iterable(flows_normal))
    df_merged_normal=pd.DataFrame.from_dict(merged_normal)

    df_merged_normal['count_dest']=df_merged_normal.apply(lambda x: time_get_count_dest(x.flow_start,x.src_ip,df_merged_normal),axis=1)
    df_merged_normal['count_src']=df_merged_normal.apply(lambda x: time_get_count_src(x.flow_start,x.dst_ip,df_merged_normal),axis=1)
    df_merged_normal['count_serv_src']=df_merged_normal.apply(lambda x: time_get_count_serv_src(x.flow_start,x.src_ip,x.dst_port,df_merged_normal),axis=1)
    df_merged_normal['count_serv_dst']=df_merged_normal.apply(lambda x: time_get_count_serv_dst(x.flow_start,x.dst_ip,x.src_port,df_merged_normal),axis=1)

    with open('flows_normal.pickle', 'wb') as handle:
        pickle.dump(df_merged_normal, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_attack=pd.read_csv("../inputs/testing.csv")
    df_attack=df_attack.sort_values(by=['tcp_stream','protocol','start_time'])
    df_attack[['flow_start']]=df_attack[['flow_start']].apply(pd.to_datetime)
    df_attack[['flow_finish']]=df_attack[['flow_finish']].apply(pd.to_datetime)
    flows_attack=generate_basic_features(df_attack)

    merged_attack = list(itertools.chain.from_iterable(flows_attack))
    df_merged_attack=pd.DataFrame.from_dict(merged_attack)

    df_merged_attack['count_dest']=df_merged_attack.apply(lambda x: time_get_count_dest(x.flow_start,x.src_ip,df_merged_attack),axis=1)
    df_merged_attack['count_src']=df_merged_attack.apply(lambda x: time_get_count_src(x.flow_start,x.dst_ip,df_merged_attack),axis=1)
    df_merged_attack['count_serv_src']=df_merged_attack.apply(lambda x: time_get_count_serv_src(x.flow_start,x.src_ip,x.dst_port,df_merged_attack),axis=1)
    df_merged_attack['count_serv_dst']=df_merged_attack.apply(lambda x: time_get_count_serv_dst(x.flow_start,x.dst_ip,x.src_port,df_merged_attack),axis=1)

    with open('flows_attack.pickle', 'wb') as handle:
            pickle.dump(df_merged_attack, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
