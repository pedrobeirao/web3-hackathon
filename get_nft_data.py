import requests
import json
import pandas as pd

def query_graph(url, query1):
    r = graph_query(url, query1)
    json_data = r.json()
    return json_data

def get_all_records(buyers):
    df_all = pd.DataFrame()
    for i in buyers:
        query_buyers = """query {
            deals (where: {buyer: \""""+str(i)+"""\"}){
                seller
                buyer
                sellTokenId
                sellToken
                buyToken
                sellAmount
                buyAmount
                price
                fee
                txHash
                blockNumber
                blockTime
                contract
            }
        }
        """
        print(i)
        r = graph_query(url, query_buyers)
        json_data = r.json()
        df_data = json_data['data']['deals']
        df_all = df_all.append(df_data)


url = 'https://api.thegraph.com/subgraphs/name/rarible/protocol'
query1 = """query {
    deals (first: 1000){
        seller
        buyer
        sellTokenId
        sellToken
        buyToken
        sellAmount
        buyAmount
        price
        fee
        txHash
        blockNumber
        blockTime
        contract
    }
}
"""

if __name__ == "__main__":
    json_data = query_graph(url, query1)
    df_data = json_data['data']['deals']
    df_deals1000 = pd.DataFrame(df_data)
    df2 = df_deals1000.sort_values(by=['blockTime'])
    buyers = df2['buyer'].unique()
    df_all = get_all_records(buyers)
    df_all.to_csv('rarible_user_data.csv')