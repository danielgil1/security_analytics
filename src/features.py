import pandas as pd
import pickle

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
                    f['fwd_packets']=f.get('fwd_packets',0)+conversation.total_packets
                    f['fwd_bytes']=f.get('fwd_bytes',0)+conversation.total_bytes
                    f['fwd_total_http_errors']=f.get('fwd_total_http_errors',0)+conversation.total_http_errors
                    f['fwd_total_failures']=f.get('fwd_total_failures',0)+conversation.total_failures
                elif (conversation.src_ip==f.get('dst_ip',"") and conversation.src_port==f.get('dst_port',"")) and (conversation.dst_ip==f.get('src_ip',"") and conversation.dst_port==f.get('src_port',"")):
                    isnewflow=False
                    f['bwd_packets']=f.get('bwd_packets',0)+conversation.total_packets
                    f['bwd_bytes']=f.get('bwd_bytes',0)+conversation.total_bytes
                    f['bwd_total_http_errors']=f.get('bwd_total_http_errors',0)+conversation.total_http_errors
                    f['bwd_total_failures']=f.get('bwd_total_failures',0)+conversation.total_failures
                    
            if isnewflow:
                new_flow={}
                
                new_flow['src_ip']=conversation.src_ip
                new_flow['dst_ip']=conversation.dst_ip
                new_flow['src_port']=conversation.src_port
                new_flow['dst_port']=conversation.dst_port
                new_flow['fwd_packets']=conversation.total_packets
                new_flow['bwd_packets']=0
                new_flow['fwd_bytes']=conversation.total_bytes
                new_flow['bwd_bytes']=0
                new_flow['fwd_total_http_errors']=conversation.total_http_errors
                new_flow['fwd_total_failures']=conversation.total_failures
                new_flow['tcp_stream']=index[0]
                new_flow['protocol']=index[1]
                flow_list.append(new_flow)
            
            
        colFlows.append(flow_list)
        
        
    return colFlows

if __name__ == "__main__":
    df_normal=pd.read_csv("../inputs/a2_feat_packet_level_normal_OCT8.csv")
    df_normal=df_normal.sort_values(by=['tcp_stream','protocol','flow_start'])
    flows_normal=generate_basic_features(df_normal)

    with open('flows_normal.pickle', 'wb') as handle:
        pickle.dump(flows_normal, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_attack=pd.read_csv("../inputs/a2_feat_packet_level_attack_OCT8.csv")
    df_attack=df_attack.sort_values(by=['tcp_stream','protocol','flow_start'])
    flows_attack=generate_basic_features(df_attack)
    with open('flows_attack.pickle', 'wb') as handle:
            pickle.dump(flows_attack, handle, protocol=pickle.HIGHEST_PROTOCOL)
    