import pandas as pd
import pickle
import itertools
import cconfig
import utils

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


def generate_extended_features(timebase,src_ip,dst_ip,src_port,dst_port,data_flow):
    
    delta=pd.Timedelta(cconfig.LAST_N_SECONDS)
    query_time=timebase-delta
    print(utils.get_time()+"Count_dest")
    count_dest=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==src_ip)].dst_ip.unique())
    print(utils.get_time()+"Count_src")
    count_src=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.dst_ip==dst_ip)].src_ip.unique())
    print(utils.get_time()+"count_serv_src")
    count_serv_src=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==src_ip) & (data_flow.dst_port==dst_port)])
    print(utils.get_time()+"count_serv_dst")
    count_serv_dst=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==dst_ip) & (data_flow.dst_port==src_port)])
    
    n_flows=cconfig.LAST_N_FLOWS
    print(utils.get_time()+"count_dest_conn")
    count_dest_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.src_ip==src_ip)].tail(n_flows).dst_ip.unique())
    print(utils.get_time()+"count_src_conn")
    count_src_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.dst_ip==dst_ip)].tail(n_flows).src_ip.unique())
    print(utils.get_time()+"count_serv_src_conn")
    count_serv_src_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.src_ip==src_ip) & (data_flow.dst_port==dst_port)].tail(n_flows))
    print(utils.get_time()+"count_serv_src_conn")
    count_serv_dst_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.src_ip==dst_ip) & (data_flow.dst_port==src_port)].tail(n_flows))
    print("done")
    return count_dest,count_src,count_serv_src,count_serv_dst,count_dest_conn,count_src_conn,count_serv_src_conn,count_serv_dst_conn

def generate_extended_features_by_loop(data_flow):
    delta=pd.Timedelta(cconfig.LAST_N_SECONDS)
    
    n_flows=cconfig.LAST_N_FLOWS
    size=len(data)
    i=1
    features_list=list()
    for index_flow,row in data_flow.iterrows():
        print(i,"/",size)
        timebase=row.flow_start
        src_ip=row.src_ip
        dst_ip=row.dst_ip
        src_port=row.src_port
        dst_port=row.dst_port

        query_time=timebase-delta
        count_dest=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==src_ip)].dst_ip.unique())
        count_src=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.dst_ip==dst_ip)].src_ip.unique())

        count_serv_src=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==src_ip) & (data_flow.dst_port==dst_port)])
        
        count_serv_dst=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.flow_start>query_time) & (data_flow.src_ip==dst_ip) & (data_flow.dst_port==src_port)])
        
        
        
        count_dest_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.src_ip==src_ip)].tail(n_flows).dst_ip.unique())
        
        count_src_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.dst_ip==dst_ip)].tail(n_flows).src_ip.unique())
        
        count_serv_src_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.src_ip==src_ip) & (data_flow.dst_port==dst_port)].tail(n_flows))
        
        count_serv_dst_conn=len(data_flow[(data_flow.flow_start<timebase) & (data_flow.src_ip==dst_ip) & (data_flow.dst_port==src_port)].tail(n_flows))
        
        i+=1
        features_list.append((count_dest,count_src,count_serv_src,count_serv_dst,count_dest_conn,count_src_conn,count_serv_src_conn,count_serv_dst_conn))
    
    with open("list_extended_features.pickle", 'wb') as handle:
            pickle.dump(features_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run(data,filename):
    print("**********************************************************")
    print("CREATING:",filename)
    data=data.sort_values(by=['tcp_stream','protocol','start_time'])
    data[['flow_start']]=data[['flow_start']].apply(pd.to_datetime)
    data[['flow_finish']]=data[['flow_finish']].apply(pd.to_datetime)
    
    print(utils.get_time()+"- Generating basic features.")
    flows=generate_basic_features(data)
    merged_flows = list(itertools.chain.from_iterable(flows))
    df_merged=pd.DataFrame.from_dict(merged_flows)
    df_merged['flow_duration']=(df_merged.flow_finish-df_merged.flow_start).dt.total_seconds() 
    
    print(utils.get_time()+"- Generating extended features.")
    extended_features=['count_dest','count_src','count_serv_src','count_serv_dst','count_dest_conn','count_src_conn','count_serv_src_conn','count_serv_dst_conn']
    #df_merged[extended_features]=df_merged.apply(lambda x: pd.Series(generate_extended_features(x.flow_start,x.src_ip,x.dst_ip,x.src_port,x.dst_port,df_merged)),axis=1)
    generate_extended_features_by_loop(df_merged)

    print(utils.get_time()+"- Saving tesing dataset into dataframe pickle.")
    with open(filename+".pickle", 'wb') as handle:
            pickle.dump(df_merged, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    pd.set_option("display.precision", 50)

    #print(utils.get_time()+"- Loading training set.")
    #df_normal=pd.read_csv("../inputs/training.csv")
    #run(df_normal,'normal_train')

    print(utils.get_time()+"- Loading testing set.")
    df_attack=pd.read_csv("../inputs/testing.csv")
    run(df_attack[:100],'attack_test')



    
    
