import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def read_expaination(path):
    '''
    输出文件信息
    :param path:
    :return:
    '''
    data = pd.read_csv(path)
    print(data.head())
    print(data.shape)
    data.info()


if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    '''缺失值判断'''
    data_train = pd.read_csv('./data/train_offline.csv')
    data_test = pd.read_csv('./data/test_offline.csv')
    print(data_train.head())
    print(data_train.isnull().sum())
    print(data_train.dtypes)
    print('-' * 250)
    print(data_test.head())
    data_test.info()
    print('=' * 250)

    '''异常值判断'''
    # 判断Coupon_id   Date_received   Discount_rate 列的空值是否一致
    print('C为空 Date_received不为空的数目:',
          data_train[(data_train['Coupon_id'].isnull()) & (data_train['Date_received'].notnull())]['User_id'].nunique())
    print('C不为空 Date_received为空的数目:',
          data_train[(data_train['Coupon_id'].notnull()) & (data_train['Date_received'].isnull())]['User_id'].nunique())
    print('C为空 Discount_rate不为空的数目:',
          data_train[(data_train['Coupon_id'].isnull()) & (data_train['Discount_rate'].notnull())]['User_id'].nunique())
    print('C不为空 Discount_rate为空的数目:',
          data_train[(data_train['Coupon_id'].notnull()) & (data_train['Discount_rate'].isnull())]['User_id'].nunique())

    print('Discount_rate', ':\t', data_train['Discount_rate'].unique().tolist())
    sns.displot(data_train['Distance'])
    plt.show()
    print('-' * 250)

    print('C为空 Date_received不为空的数目:',
          data_test[(data_test['Coupon_id'].isnull()) & (data_test['Date_received'].notnull())]['User_id'].nunique())
    print('C不为空 Date_received为空的数目:',
          data_test[(data_test['Coupon_id'].notnull()) & (data_test['Date_received'].isnull())]['User_id'].nunique())
    print('C为空 Discount_rate不为空的数目:',
          data_test[(data_test['Coupon_id'].isnull()) & (data_test['Discount_rate'].notnull())]['User_id'].nunique())
    print('C不为空 Discount_rate为空的数目:',
          data_test[(data_test['Coupon_id'].notnull()) & (data_test['Discount_rate'].isnull())]['User_id'].nunique())

    print('Discount_rate', ':\t', data_test['Discount_rate'].unique().tolist())
    sns.displot(data_test['Distance'])
    plt.show()
