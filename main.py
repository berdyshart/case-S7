import pandas as pd
import matplotlib.pyplot as plt
from unicodedata import category


def graph(obj, plt_title="no title", plt_xlabel="D", plt_ylabel="qty", plt_legend="category"):
    obj.plot(figsize=(12, 6), marker='o', markersize = 2, linestyle='')
    plt.title(plt_title)
    plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    plt.grid(True)
    plt.legend(title=plt_legend)
    plt.show()


def volatility_of_price_fun(df):
    df = df.copy()
    price_volatility = df.groupby('product_category')['price'].std() / df.groupby('product_category')['price'].mean()
    return price_volatility


def lead_time_fun(df):
    df = df.copy()
    df = df[df['product_category'] == 2]
    df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days
    avg_lead_time = df.groupby('product_category')['lead_time'].mean()
    return avg_lead_time


def std_lead_time_fun(df):
    df = df.copy()
    df = df[df['product_category'] == 2]
    df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days
    std_lead_time = df.groupby('product_category')['lead_time'].std()
    return std_lead_time


def fillrate_fun(df):
    df = df.copy()
    df = df[df['product_category'] == 2]
    delivery_relatability = df.groupby('product_category')['valid_delivered_qty'].mean() / df.groupby('product_category')['qty'].mean()
    return delivery_relatability


def seasonality_fun(df):
    df = df.copy()
    df.set_index('order_date', inplace=True)
    seasonality = df.groupby('product_category')['qty'].resample('ME').count().unstack(level=0)

    return seasonality


def consumption_fun(df_consumption, category=-1):
    df = df_consumption.copy()
    df.set_index('consumtion_date', inplace=True)
    if category != -1:
        df = df[df['product_category'] == category]
    consumption = df.groupby('product_category')['qty'].resample('D').sum().unstack(level=0)

    return consumption


def consumption_average_fun(df_consumption):
    df = df_consumption.copy()
    df = df[df['product_category'] == 0]
    df.set_index('order_date', inplace=True)
    consumption_average = df.groupby('product_category')['qty'].mean()

    return consumption_average


def consumption_fun_for_cat1(df_consumption):
    df = df_consumption.copy()
    df.set_index('consumtion_date', inplace=True)
    df = df[df['product_category'] == 1]
    consumption = df.groupby('product_category')['qty'].resample('ME').count().unstack(level=0)

    consumption.plot(figsize=(12, 6), marker='o')
    plt.title('Сезонность потребления по категориям (qty по месяцам)')
    plt.xlabel('Дата')
    plt.ylabel('Количество потребленное')
    plt.grid(True)
    plt.legend(title='Категории')
    plt.show()


def abc_analysis_by_year(df):
    result = []

    def classify_abc(percentage):
        if percentage <= 80:
            return 'A'
        elif percentage <= 95:
            return 'B'
        else:
            return 'C'


    df['year'] = df['order_date'].dt.year
    abc_df_temp = df.groupby(['year', 'product_category'])['amount'].sum().reset_index()
    years = sorted(df['year'].unique())

    for year in years:
        abc_df = abc_df_temp[abc_df_temp['year'] == year]
        abc_df = abc_df.sort_values(by='amount', ascending=False)

        abc_df['cum_sum'] = abc_df['amount'].cumsum()
        abc_df['cum_perc'] = abc_df['cum_sum'] / abc_df['amount'].sum() * 100

        abc_df['class'] = abc_df['cum_perc'].apply(classify_abc)

        result.append(abc_df)
        abc_df = abc_df_temp[abc_df_temp['year'] == year]

    return result


def xyz_analisis_by_year(df_consumption):
    df_consumption['year'] = df_consumption['consumtion_date'].dt.year
    years = sorted(df_consumption['year'].unique())
    result = []

    for year in years:
        df_consumption_local = df_consumption[df_consumption['year'] == year]
        xyz_matrix = df_consumption_local.pivot_table(
            index='product_category',
            columns=df_consumption_local['consumtion_date'].dt.to_period('M'),
            values='qty',
            aggfunc='sum'
        ).fillna(0)

        mean = xyz_matrix.mean(axis=1)
        std = xyz_matrix.std(axis=1)
        xyz_df = pd.DataFrame({'cv': std / mean},
                              index=xyz_matrix.index).reset_index()

        xyz_df['xyz_class'] = pd.cut(xyz_df['cv'],
                                     bins=[0, 0.1, 0.25, float('inf')],
                                     labels=['X', 'Y', 'Z'])

        result.append(xyz_df)

    return result


def prices(df, category = -1):
    df = df.copy()
    df.set_index('order_date', inplace=True)
    if category != -1:
        df = df[df['product_category'] == category]
    prices = df.groupby('product_category')['price'].resample(
        'D').mean().unstack(level=0)

    return prices


def zero_point_percentage(df_consumption):
    '''
    Function, calculating the percentage of zero points in data frame
    :param df_consumption:
    :return:
    '''
    df = df_consumption.copy()



def main():
    # импорт данных
    df = pd.read_excel('C:\\Users\\berdy\\OneDrive\\Рабочий стол\\final_orders_train.csv.xlsx')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df = df[df['qty'] > 0].dropna(subset=['qty', 'product_category'])

    df_consumption = pd.read_excel('C:\\Users\\berdy\\OneDrive\\Рабочий стол\\final_consumtion_train.csv.xlsx')
    df_consumption['consumtion_date'] = pd.to_datetime(df_consumption['consumtion_date'])
    df_consumption = df_consumption[df_consumption['qty'] > 0].dropna(subset=['qty', 'product_category', 'consumtion_date'])

    # for category in [0, 1, 2, 3, 4]:
    # # graph(consumption_fun(df_consumption, category = category), plt_title=f"Потребление для категории {category}", plt_xlabel="D")
    #     graph(prices(df, category = category), plt_title=f"Цены категории {category}", plt_xlabel="D", plt_ylabel="Price")
    print(lead_time_fun(df), std_lead_time_fun(df), fillrate_fun(df))

main()
