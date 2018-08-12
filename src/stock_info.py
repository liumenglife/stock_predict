import requests
from bs4 import BeautifulSoup
import pandas as pd
# pd.core.common.is_list_like = pd.api.types.is_list_like
# from pandas_datareader.google.daily import GoogleDailyReader
# import pandas_datareader as pdr
from datetime import datetime


class StockInfo:
    def __init__(self):
        self.code_df = self._init()

    def _init(self):
        code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[
            0]
        # 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌
        code_df.종목코드 = code_df.종목코드.map('{:06d}'.format)
        # 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다.
        code_df = code_df[['회사명', '종목코드']]  # 한글로된 컬럼명을 영어로 바꿔준다.
        code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})

        # print(code_df.shape[0])
        # print(code_df.head())

        return code_df

    # def get_stock_list(self, from_internet=False):
    #     """
    #     Currently not working, url should be properly replaced
    #
    #     :return:
    #     """
    #
    #     stock_list = []
    #
    #     if from_internet:
    #         url = 'http://datamall.koscom.co.kr/servlet/infoService/SearchIssue'
    #         source_code = requests.get(url)
    #         plain_text = source_code.text
    #         soup = BeautifulSoup(plain_text, 'lxml')
    #
    #         select = soup.find_all('select', attrs={'name': 'stockCodeList'})
    #         for l_select in select:
    #             for l_option in l_select.find_all('option'):
    #                 stock = {}
    #                 if not "폐지" in l_option.text:
    #                     stock['code'] = l_option.text[1:7]
    #                     stock['name'] = l_option.text[8:]
    #                     stock['full_code'] = l_option.get('value')
    #                     stock_list.append(stock)
    #
    #     else:
    #         df = pd.read_excel('./external_info/stock_list.xlsx')
    #         for code, name in zip(df['CODE'], df['종목명']):
    #             stock = dict()
    #             stock['code'] = code[1:7]
    #             stock['name'] = name
    #             stock['full_code'] = code
    #             stock_list.append(stock)
    #
    #     return stock_list

    def get_stock_data(self, item_name):
        t = datetime.now()
        print('Crawling Company Name:', item_name)
        code = self.code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)

        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
        print("요청 URL = {}".format(url))

        df = pd.DataFrame()

        pg_url = '{url}&page={page}'.format(url=url, page=1)
        d = pd.read_html(pg_url, header=0)[0]
        d = d.dropna()
        df = df.append(d, ignore_index=True)

        for page in range(2, 1000):
            # print(page)
            # if page % 50 == 0:
            #     print(page)
            pg_url = '{url}&page={page}'.format(url=url, page=page)
            d = pd.read_html(pg_url, header=0)[0]
            d = d.dropna()

            # if d['날짜'].shape != 0 and d['날짜'][1] in df['날짜'].tolist():
            if d['날짜'][1] in df['날짜'].tolist():
                print('Break page:', page)
                break
            df = df.append(d, ignore_index=True)

        df['name'] = pd.Series([item_name] * len(df['날짜']), index=df.index)
        df = df.rename(columns={'날짜': 'date', '종가': 'final_price', '전일비': 'compare_to_prior', '시가': 'start_price',
                                '고가': 'highest_price', '저가': 'lowest_price', '거래량': 'num_of_traded'})
        t = datetime.now() - t
        print('Elapsed time:', t)

        return df


# KRX:code[1:7]


if __name__ == '__main__':
    stock_info = StockInfo()
    # stock_list = stock_info.get_stock_list()
    # print('Stock List:', stock_list)

    # print(stock_list[0]['name'])
    company_name = stock_info.code_df.head(stock_info.code_df.shape[0])
    # print(company_name['name'][0])
    print('Number of companies:', len(company_name['name']))

    # print(type(company_name))
    i = 1
    all_data = None
    a = datetime.now()
    for idx, name in enumerate(company_name['name']):
        if idx < 1753:
            continue
        elif idx == 325:
            continue
        print('idx:', idx)
        data = stock_info.get_stock_data(item_name=name)

        if idx == 0:
            all_data = data
        else:
            # all_data.append(data, ignore_index=True)
            all_data = pd.concat([all_data, data])
        print('data shape:', data.shape[0])
        print('all data shape', all_data.shape[0])
        print('')

        if idx % 100 == 0:
            all_data.to_csv('/Volumes/osx_sub/데이터/주식데이터/2018_07_28_stock_data.csv')
        # print(all_data.columns.values)
        # if idx == i:
        #     break

    print('Crawling data finished. Total elapsed time:', datetime.now() - a)

    all_data.to_csv('/Volumes/osx_sub/데이터/주식데이터/2018_07_28_stock_data.csv')
    print('Data saved!!')





# a = pdr.DataReader(stock_list[0]['code'], 'google', '2018-01-01', '2018-07-12')
# print(a)