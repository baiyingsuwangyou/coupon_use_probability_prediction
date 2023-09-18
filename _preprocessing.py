import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


class preprocessing:
    def __init__(self, root_path):
        self.root_path = root_path

    def read_and_null_process(self, is_test):
        '''
        读取数据和空值处理
        :param is_test: 是否为测试集
        :return: 数据
        '''
        # 读取数据
        if is_test:
            data = pd.read_csv(os.path.join(self.root_path, 'test_offline.csv'), parse_dates=['Date_received'])
        else:
            data = pd.read_csv(os.path.join(self.root_path, 'train_offline.csv'), parse_dates=['Date_received', 'Date'])

        # 空值处理
        data['Distance'].fillna(11, inplace=True)
        data['Distance'] = data['Distance'].astype(int)
        data['Coupon_id'].fillna(0, inplace=True)
        data['Coupon_id'] = data['Coupon_id'].astype(int)

        data['Discount_rate'].fillna(0, inplace=True)

        print(data.head())
        data.info()
        return data

    def label_precess(self, data, is_test):
        '''
        添加标签列，并将Discount_rate列变为数值类型
        :param data:
        :param is_test:
        :return: data_new
        '''
        # Discount_rate 列 处理
        data['Discount_rate'] = data['Discount_rate'].map(
            lambda x: 1 - int(x.split(':', 1)[1]) / int(x.split(':', 1)[0]) if ':' in str(x) else x)
        data['Discount_rate'] = data['Discount_rate'].astype(float)

        # data['day'] = data['Date_received'].dt.day.fillna(0)
        # data['day'] = data['day'].astype(int)

        # label     0 :未消费   1 :普通消费     2 :未15天内使用优惠券消费    3 :15天内使用优惠券消费
        def set_label(row):
            if pd.isnull(row['Date']):
                return 0
            elif row['Coupon_id'] == 0:
                return 1
            else:
                data_diff = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'],
                                                                                          format='%Y%m%d')
                if data_diff > pd.Timedelta(15, 'D'):
                    return 2
                else:
                    return 3

        if not is_test:
            data['label'] = data.apply(set_label, axis=1)

        print(data.head())
        data.info()
        return data


if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    if not os.path.exists('./preprocess_data'):
        os.makedirs('./preprocess_data')

    root_path = './data'
    n = 100000
    p = preprocessing(root_path)

    train = p.read_and_null_process(is_test=False).sample(n, random_state=0)
    train.reset_index(drop=True, inplace=True)
    train = p.label_precess(train, is_test=False)
    train.to_csv('./preprocess_data/train_p.csv', index=False)

    test = p.read_and_null_process(is_test=True)
    test = p.label_precess(test, is_test=True)
    test.to_csv('./preprocess_data/test_p.csv', index=False)
