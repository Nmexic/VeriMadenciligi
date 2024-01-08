
# CUSTOMER LIFETIME VALUE (Müşteri Yaşam Boyu Değeri)


# 1. Veri Hazırlama
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Dataset Değişkenleri
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel('online_retail_II.xlsx', sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T
df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate

cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10

cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

cltv_c["CLTV"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

# cltv_c.sort_values(by="CLTV", ascending=False).head()

cltv_c["segment"] = pd.qcut(cltv_c["CLTV"], 4, labels=["D", "C", "B", "A"])

cltv_c.groupby("segment").agg({"count", "mean", "sum"})

cltv_c.to_excel("cltv_c.xlsx")


#################################################################################
# CLTV PREDICTION
#################################################################################


import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel('online_retail_II.xlsx', sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.isnull().sum()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2011, 12, 11)

cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# Recency (Recency):
#   Müşterinin son satın alma işleminden bu yana geçen zamanı ifade eder.

# T (Tenure):
#   Müşterinin ilk satın alma işleminden bu yana geçen toplam süreyi ifade eder.
#   Recency ile birlikte kullanılarak müşterinin ne kadar süredir aktif olduğunu belirler.

# Frequency (Frequency):
#   Müşterinin satın alma işlemlerinin toplam sayısını ifade eder. Yani müşterinin kaç defa alışveriş yaptığıdır.

# Monetary (Monetary Value):
#   Müşterinin toplam harcamasını ifade eder.
#   Frequency ile birlikte kullanılarak ortalama satın alma değeri hesaplanır.

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# BG NBD MODELİNİN KURULMASI

# Bayesian bir istatistik modeli olan Beta Geometric/Negative Binomial Distribution (BG/NBD) modelini kullanarak
# müşteri satın alma davranışını modeller ve müşteri yaşam boyu değerini tahmin eder.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


cltv_df["exp_purc_1_week"] = bgf.predict(1,
                                         cltv_df['frequency'],
                                         cltv_df['recency'],
                                         cltv_df['T'])

cltv_df["exp_purc_1_month"] = bgf.predict(4,
                                          cltv_df['frequency'],
                                          cltv_df['recency'],
                                          cltv_df['T'])

cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])


cltv_df.to_excel("cltv_final.xlsx")
plot_period_transactions(bgf)
plt.show()
