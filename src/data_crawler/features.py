import numpy as np
import pandas as pd


class Features:

    @staticmethod
    def fnMACD(m_Df, m_NumFast=12, m_NumSlow=26, m_NumSignal=9):
        m_Df['EMAFast'] = m_Df['final_price'].ewm(span=m_NumFast, min_periods=m_NumFast - 1).mean()
        m_Df['EMASlow'] = m_Df['final_price'].ewm(span=m_NumSlow, min_periods=m_NumSlow - 1).mean()
        m_Df['MACD'] = m_Df['EMAFast'] - m_Df['EMASlow']
        m_Df['MACDSignal'] = m_Df['MACD'].ewm(span=m_NumSignal, min_periods=m_NumSignal - 1).mean()
        m_Df['MACDDiff'] = m_Df['MACD'] - m_Df['MACDSignal']
        return m_Df

    # df_new = pd.DataFrame()
    #
    # for idx, name in enumerate(df.name.unique()):
    #     if idx == 2:
    #         break
    #     df_new = pd.concat([df_new, fnMACD(df[df['name'] == name].sort_values(by=['date']).reset_index(drop=True))])
    #
    # df_new = df_new.reset_index(drop=True)

    @staticmethod
    def fnBolingerBand(m_DF, n=20, k=2):
        m_DF['20d_ma'] = pd.rolling_mean(m_DF['final_price'], window=n)
        m_DF['Bol_upper'] = pd.rolling_mean(m_DF['final_price'], window=n) + k * pd.rolling_std(m_DF['final_price'], n,
                                                                                                min_periods=n)
        m_DF['Bol_lower'] = pd.rolling_mean(m_DF['final_price'], window=n) - k * pd.rolling_std(m_DF['final_price'], n,
                                                                                                min_periods=n)
        # m_DF['20d_ma'] = m_DF['final_price'].rolling(n).mean()
        # m_DF['Bol_upper'] = m_DF['final_price'].rolling(n).mean() + k * pd.rolling_std(m_DF['final_price'], n,
        #                                                                                         min_periods=n)
        # m_DF['Bol_lower'] = m_DF['final_price'].rolling(n).mean() - k * pd.rolling_std(m_DF['final_price'], n,
        #                                                                                         min_periods=n)
        return m_DF

    @staticmethod
    def fnRSI(m_Df, m_N=14):
        U = np.where(m_Df['final_price'].diff(1) > 0, m_Df['final_price'].diff(1), 0)
        D = np.where(m_Df['final_price'].diff(1) < 0, m_Df['final_price'].diff(1) * (-1), 0)

        AU = pd.DataFrame(U).rolling(window=m_N, min_periods=m_N).mean()
        AD = pd.DataFrame(D).rolling(window=m_N, min_periods=m_N).mean()
        RSI = AU.div(AD + AU) * 100

        m_Df['RSI'] = RSI
        return m_Df

    @staticmethod
    def fnStoch(m_Df, n=14):  # price: 종가(시간 오름차순), n: 기간
        sz = len(m_Df['final_price'])
        if sz < n:
            # show error message
            raise SystemExit('입력값이 기간보다 작음')
        tempSto_K = []
        for i in range(sz):
            if i >= n - 1:
                tempUp = m_Df['final_price'][i] - min(m_Df['lowest_price'][i - n + 1:i + 1])
                tempDown = max(m_Df['highest_price'][i - n + 1:i + 1]) - min(m_Df['lowest_price'][i - n + 1:i + 1])
                tempSto_K.append(tempUp / tempDown)
            else:
                tempSto_K.append(0)  # n보다 작은 초기값은 0 설정
        m_Df['Sto_K'] = pd.Series(tempSto_K, index=m_Df.index)

        m_Df['Sto_D'] = pd.Series(pd.rolling_mean(m_Df['Sto_K'], 3))
        m_Df['Sto_SlowD'] = pd.Series(pd.rolling_mean(m_Df['Sto_D'], 3))
        # m_Df['Sto_D'] = pd.Series(m_Df['Sto_K'].rolling(3).mean())
        # m_Df['Sto_SlowD'] = pd.Series(m_Df['Sto_D'].rolling(3).mean())

        return m_Df

    @staticmethod
    def fnMA(m_Df, m_N=list(), m_ColumnName='final_price'):
        all_MA = list()
        if m_ColumnName in m_Df.columns:
            for num in m_N:
                #             MA = pd.Series(pd.rolling_mean(m_Df[m_ColumnName], num), name = 'MA' + str(num))
                #             m_Df = m_Df.join(MA)
                MA = pd.Series.rolling(m_Df[m_ColumnName], window=num, center=False).mean()
                m_Df['MA' + str(num)] = MA

                all_MA.append(MA)

            for i in range(len(all_MA)):
                if i + 1 == len(all_MA):
                    break

                for i2 in range(i + 1, len(all_MA)):
                    m_Df['SignalMA' + str(m_N[i]) + '_' + str(m_N[i2])] = all_MA[i] - all_MA[i2]

        else:
            raise ("You didn't input a Column Name")
        return m_Df

    @staticmethod
    def change_prior_to(m_Df):

        m_Df['compare_to_prior'] = m_Df['final_price'].diff(1)
        m_Df['percent'] = (m_Df['final_price'] * 100 / (m_Df['final_price'] - m_Df['compare_to_prior']) - 100).round(2)

        return m_Df

