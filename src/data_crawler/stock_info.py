
import pandas as pd
# pd.core.common.is_list_like = pd.api.types.is_list_like
# from pandas_datareader.google.daily import GoogleDailyReader
# import pandas_datareader as pdr
from datetime import datetime
import numpy as np
from os.path import join, exists, dirname
from os import makedirs


class StockInfo:
    def __init__(self):
        self.code_df = self._retrieve_company_list()

    def _retrieve_company_list(self):
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

    def get_stock_data(self, item_name, by_year=list(), recent_n_data=list()):
        t = datetime.now()
        print('Crawling Company Name:', item_name)
        code = self.code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)

        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
        print("요청 URL = {}".format(url))

        yrs = ""
        min_year = -1

        target_date = -1
        num_seq = -1

        if len(recent_n_data) == 2:
            target_date = str(recent_n_data[0])
            target_date = target_date[:4] + '.' + target_date[4:6] + '.' + target_date[6:8]  # yyyy.mm.dd
            num_seq = recent_n_data[1]

        if len(by_year) != 0 and target_date == -1:
            yrs = "|".join(str(yr) for yr in by_year)
            min_year = sorted(by_year)[0]
            print('Collecting years... ', by_year)
            # print('Mininum Year: %i' % min_year)

        df = pd.DataFrame()

        for page in range(1, 1000):
            # print(page)
            # if page % 50 == 0:
            #     print(page)
            pg_url = '{url}&page={page}'.format(url=url, page=page)
            d = pd.read_html(pg_url, header=0)[0]
            d = d.dropna()

            # Check if it crawls duplicate
            if df.shape[0] != 0 and d['날짜'][1] in df['날짜'].tolist():
                # Max page reached
                print('Break page:', page)
                break

            # If crawling requires by only inquired target date and its sequence length
            if target_date != -1:
                if df.shape[0] == 0:
                    d = d[d['날짜'].str.contains(target_date)]
                    if d.shape[0] == 0:
                        continue

                df = df.append(d)

                if df.shape[0] >= num_seq:
                    df = df.head(num_seq)
                    break

            # If crawling requires by only inquired years
            if min_year != -1 and target_date == -1 and min_year > int(d['날짜'][d.shape[0] - 1][:4]):
                print('Found minimum year, break page:', page)
                break

            df = df.append(d, ignore_index=True)

        df['name'] = pd.Series([item_name] * len(df['날짜']), index=df.index)
        df = df.rename(columns={'날짜': 'date', '종가': 'final_price', '전일비': 'compare_to_prior', '시가': 'start_price',
                                '고가': 'highest_price', '저가': 'lowest_price', '거래량': 'num_of_traded'})

        # Only by year
        if min_year != -1:
            df = df[df['date'].str.contains(yrs)]

        t = datetime.now() - t
        print('Elapsed time:', t)

        return df

    def collect_data(self, output_path, company_name=list(), by_year=list(), recent_n_data=list()):
        """

        :param output_path:
        :param company_name: Name of the company to search
        :param by_year: [year1, year2, ..., year n]; all year should be number
        :param recent_n_data: [target_date in yyyymmdd, number of previous date to be crawled]
        :return:
        """

        company_list = self.code_df.head(self.code_df.shape[0])

        if len(company_name) == 0:
            company_name = company_list['name']

        print('Number of companies:', len(company_name))

        i = 1
        all_data = None
        a = datetime.now()

        # crawling failed company list
        fcl = list()

        for idx, name in enumerate(company_name):
            try:
                # if idx != 325:
                #     continue

                print('idx:', idx)
                data = self.get_stock_data(item_name=name, by_year=by_year, recent_n_data=recent_n_data)

                if idx == 0:
                    all_data = data

                else:
                    # all_data.append(data, ignore_index=True)
                    all_data = pd.concat([all_data, data])
                print('data shape:', data.shape[0])
                print('all data shape', all_data.shape[0])
                print('')


                # all_data.to_csv(output_path)
                # print(all_data.columns.values) # show column list
                # if idx == i:
                #     break

            except Exception as e:
                fcl.append(name)


        print('Crawling data finished. Total elapsed time:', datetime.now() - a)
        print('Crawling failed company list:', fcl)

        if not exists(dirname(output_path)):
            makedirs(dirname(output_path))

        all_data.to_csv(output_path)
        print('Data saved!!')

    def split_data_by_company(self, input_path, output_path):

        df = pd.read_csv(input_path)

        if not exists(output_path):
            makedirs(output_path)

        for idx, company_name in enumerate(df.name.unique()):
            print(idx, company_name)
            a = df.loc[df['name'] == company_name]
            a = a.loc[:, a.columns != 'name']
            np.save(join(output_path, company_name), a)

        print('All data saved by company name. Length of company:', len(df.name.unique()))



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