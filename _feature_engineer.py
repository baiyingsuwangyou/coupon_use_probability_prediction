import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


class feature_engineer:
    def __init__(self, root_path):
        self.root_path = root_path

    def process(self, is_test):
        # 读取数据
        if is_test:
            data = pd.read_csv(os.path.join(self.root_path, 'test_p.csv'), parse_dates=['Date_received'])
        else:
            data = pd.read_csv(os.path.join(self.root_path, 'train_p.csv'), parse_dates=['Date_received', 'Date'])
            # print(data['label'].unique().tolist())

        # 数据准备
        normal_consume_1 = data[data['Date_received'].isnull()]  # 普通消费(无优惠券消费) 1
        coupon_have_023 = data[data['Date_received'].notnull()]  # 领取优惠券 023
        '''# 添加新特征'''

        def x1(data, used_data, used_name, new_name):
            temp = used_data.groupby(used_name).size().reset_index(name=new_name)
            data_p = pd.merge(data, temp, how='left', on=used_name)
            return data_p

        def xy1(data, used_data, used_name_x, used_name_y, new_name):
            temp = used_data.groupby([used_name_x, used_name_y]).size().reset_index(name=new_name)
            data_p = pd.merge(data, temp, how='left', on=[used_name_x, used_name_y])
            return data_p

        def xy2(data, used_data, used_name_x, used_name_y, new_name):
            temp = used_data.groupby([used_name_x, used_name_y]).size().reset_index()
            temp = temp.groupby(used_name_x).size().reset_index(name=new_name)
            data_p = pd.merge(data, temp, how='left', on=used_name_x)
            return data_p

        def xyz(data, used_data, used_name_x, used_name_y, used_name_z, new_name):
            temp = used_data.groupby([used_name_x, used_name_y, used_name_z]).size().reset_index(name=new_name)
            data_p = pd.merge(data, temp, how='left', on=[used_name_x, used_name_y, used_name_z])
            return data_p

        '''User_id_feature'''

        # 该用户普通消费次数
        data_addition = x1(data, normal_consume_1, 'User_id', 'u0')
        # 该用户领取优惠券次数
        data_addition = x1(data_addition, coupon_have_023, 'User_id', 'u1')
        # 总数据中该用户出现次数
        data_addition = x1(data_addition, data, 'User_id', 'u2')

        # 该用户消费优惠券的平均折率
        temp = coupon_have_023.groupby('User_id')['Discount_rate'].mean().reset_index(name='u3')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')
        # 该用户消费优惠券的最高折率
        temp = coupon_have_023.groupby('User_id')['Discount_rate'].max().reset_index(name='u4')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')
        # 该用户消费优惠券的最低折率
        temp = coupon_have_023.groupby('User_id')['Discount_rate'].min().reset_index(name='u5')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')

        # 该用户与消费的商家平均距离asnot
        temp1 = coupon_have_023[coupon_have_023['Distance'] != 11]  # Distance不为空的数据
        temp = temp1.groupby('User_id')['Distance'].mean().reset_index(name='u6')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')
        # 该用户与消费的商家最大距离
        temp = temp1.groupby('User_id')['Distance'].max().reset_index(name='u7')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')
        # 该用户与消费的商家最小距离
        temp = temp1.groupby('User_id')['Distance'].min().reset_index(name='u8')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')

        # 该用户与消费的商家平均距离asnot
        temp1 = data[data['Distance'] != 11]  # Distance不为空的数据
        temp = temp1.groupby('User_id')['Distance'].mean().reset_index(name='u9')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')
        # 该用户与消费的商家最大距离
        temp = temp1.groupby('User_id')['Distance'].max().reset_index(name='u10')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')
        # 该用户与消费的商家最小距离
        temp = temp1.groupby('User_id')['Distance'].min().reset_index(name='u11')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')

        # 该用户与消费的商家平均距离asnot
        temp1 = normal_consume_1[normal_consume_1['Distance'] != 11]  # Distance不为空的数据
        temp = temp1.groupby('User_id')['Distance'].mean().reset_index(name='u12')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')
        # 该用户与消费的商家最大距离
        temp = temp1.groupby('User_id')['Distance'].max().reset_index(name='u13')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')
        # 该用户与消费的商家最小距离
        temp = temp1.groupby('User_id')['Distance'].min().reset_index(name='u14')
        data_addition = pd.merge(data_addition, temp, how='left', on='User_id')

        # 该用户不同优惠券的领取次数
        data_addition = xy1(data_addition, coupon_have_023, 'User_id', 'Discount_rate', 'u15')

        # 该用户在特定商户使用或领取优惠券的次数
        data_addition = xy1(data_addition, data, 'User_id', 'Merchant_id', 'u16')
        # 该用户在特定商户领取优惠券次数
        data_addition = xy1(data_addition, coupon_have_023, 'User_id', 'Merchant_id', 'u17')
        # 该用户在特定商户普通消费次数
        data_addition = xy1(data_addition, normal_consume_1, 'User_id', 'Merchant_id', 'u18')

        # 该用户领取特定优惠券的次数
        data_addition = xy1(data_addition, coupon_have_023, 'User_id', 'Coupon_id', 'u19')

        # 该用户该天领取优惠券的数目
        data_addition = xy1(data_addition, coupon_have_023, 'User_id', 'Date_received', 'u20')

        # 该用户领取过的优惠券种类数目
        data_addition = xy2(data_addition, coupon_have_023, 'User_id', 'Coupon_id', 'u21')

        # 该用户领取优惠券的不同商户数量
        data_addition = xy2(data_addition, coupon_have_023, 'User_id', 'Merchant_id', 'u22')

        # 该用户当天领取特定优惠券的数目
        data_addition = xyz(data_addition, coupon_have_023, 'User_id', 'Coupon_id', 'Date_received', 'u23')

        # 该用户当天在特定商户领取优惠券的次数
        data_addition = xyz(data_addition, coupon_have_023, 'User_id', 'Merchant_id', 'Date_received', 'u24')

        '''Merchant_id_feature'''

        # 用户在该商户正常消费的次数
        data_addition = x1(data_addition, normal_consume_1, 'Merchant_id', 'm0')
        # 用户在该商户领取优惠券的次数
        data_addition = x1(data_addition, coupon_have_023, 'Merchant_id', 'm1')
        # 总数据中该商户出现次数
        data_addition = x1(data_addition, data, 'Merchant_id', 'm2')

        # 该商户当天优惠券领取次数
        data_addition = xy1(data_addition, coupon_have_023, 'Merchant_id', 'Date_received', 'm3')

        # 该商户优惠券消费的平均折率
        temp = coupon_have_023.groupby('Merchant_id')['Discount_rate'].mean().reset_index(name='m4')
        data_addition = pd.merge(data_addition, temp, how='left', on='Merchant_id')
        # 该商户优惠券消费的最大折率
        temp = coupon_have_023.groupby('Merchant_id')['Discount_rate'].max().reset_index(name='m5')
        data_addition = pd.merge(data_addition, temp, how='left', on='Merchant_id')
        # 该商户优惠券消费的最小折率
        temp = coupon_have_023.groupby('Merchant_id')['Discount_rate'].min().reset_index(name='m6')
        data_addition = pd.merge(data_addition, temp, how='left', on='Merchant_id')

        # 该商户被领取的特定优惠券数目
        data_addition = xy1(data_addition, coupon_have_023, 'Merchant_id', 'Coupon_id', 'm7')

        # 该商户领取过优惠券的不同用户数量
        data_addition = xy2(data_addition, coupon_have_023, 'Merchant_id', 'User_id', 'm8')

        # 该商户领取过的优惠券种类数目
        data_addition = xy2(data_addition, coupon_have_023, 'Merchant_id', 'Coupon_id', 'm9')

        '''Coupon_id_feature'''

        # 该优惠券共发行的数目
        data_addition = x1(data_addition, coupon_have_023, 'Coupon_id', 'c0')

        # 该优惠券在当天发行的数目
        data_addition = xy1(data_addition, coupon_have_023, 'Coupon_id', 'Date_received', 'c1')

        # 该折扣优惠券领取次数
        data_addition = x1(data_addition, coupon_have_023, 'Discount_rate', 'c2')

        if is_test:
            data_addition = data_addition.drop(
                columns=['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received'], axis=1)
        else:
            data_addition = data_addition.drop(
                columns=['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received', 'Date'], axis=1)
        data_addition = data_addition.fillna(0)

        print(data_addition.head())
        data_addition.info()
        return data_addition


if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    f = feature_engineer('./preprocess_data')
    train = f.process(is_test=False)
    train.to_csv('./preprocess_data/train_f.csv', index=False)
    # corr = train.corr()
    # print(corr)
    # corr.to_csv('./corr_data.csv')

    test = f.process(is_test=True)
    test.to_csv('./preprocess_data/test_f.csv', index=False)
