import pandas as pd
import matplotlib.pyplot as plt


def volatility():
    global DF
    df = DF.copy()
    price_volatility = df.groupby('product_category')['price'].std() / df.groupby('product_category')['price'].mean()
    return price_volatility


def lead_time():
    global DF
    df = DF.copy()
    # df.reset_index(inplace=True)
    df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days
    avg_lead_time = df.groupby('product_category')['lead_time'].mean()
    return avg_lead_time


def fillrate():
    global DF
    df = DF.copy()
    delivery_relatability = df.groupby('product_category')['valid_delivered_qty'].mean() / df.groupby('product_category')['qty'].mean()
    return delivery_relatability


def season():
    global DF
    df = DF.copy()
    df.set_index('order_date', inplace=True)
    seasonality = df.groupby('product_category')['price'].resample('ME').count().unstack(level=0)
    seasonality.plot(figsize=(12, 6), marker='o')
    plt.title('Сезонность закупок по категориям (объем закупок по месяцам)')
    plt.xlabel('Дата')
    plt.ylabel('Объем закупок')
    plt.grid(True)
    plt.legend(title='Категории')
    plt.show()


def consup():
    global DF_consump
    df = DF_consump.copy()
    df.set_index('consumtion_date', inplace=True)
    comsuption = df.groupby('product_category')['qty'].resample('ME').count().unstack(level=0)
    comsuption.plot(figsize=(12, 6), marker='o')
    plt.title('Сезонность потребления по категориям (qty по месяцам)')
    plt.xlabel('Дата')
    plt.ylabel('Количество потребленное')
    plt.grid(True)
    plt.legend(title='Категории')
    plt.show()


def abc_analisys():
    result = []

    def classify_abc(percentage):
        if percentage <= 80:
            return 'A'
        elif percentage <= 95:
            return 'B'
        else:
            return 'C'

    global DF
    DF['year'] = DF['order_date'].dt.year
    abc_df_temp = DF.groupby(['year', 'product_category'])['amount'].sum().reset_index()
    years = sorted(DF['year'].unique())

    for year in years:
        abc_df = abc_df_temp[abc_df_temp['year'] == year]
        abc_df = abc_df.sort_values(by='amount', ascending=False)

        abc_df['cum_sum'] = abc_df['amount'].cumsum()
        abc_df['cum_perc'] = abc_df['cum_sum'] / abc_df['amount'].sum() * 100

        abc_df['class'] = abc_df['cum_perc'].apply(classify_abc)

        result.append(abc_df)
        abc_df = abc_df_temp[abc_df_temp['year'] == year]
    return result


def xyz_analisys():
    global DF_consump
    DF_consump['year'] = DF_consump['consumtion_date'].dt.year
    years = sorted(DF_consump['year'].unique())
    result = []

    for year in years:
        DF_consump_local = DF_consump[DF_consump['year'] == year]
        xyz_matrix = DF_consump_local.pivot_table(
            index='product_category',
            columns=DF_consump_local['consumtion_date'].dt.to_period('M'),
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


# импорт данных
DF = pd.read_excel('C:\\Users\\berdy\\OneDrive\\Рабочий стол\\final_orders_train.csv.xlsx')
DF['order_date'] = pd.to_datetime(DF['order_date'])
DF['delivery_date'] = pd.to_datetime(DF['delivery_date'])
DF = DF[DF['qty'] > 0]
DF = DF.dropna(subset=['qty', 'product_category'])

DF_consump = pd.read_excel('C:\\Users\\berdy\\OneDrive\\Рабочий стол\\final_consumtion_train.csv.xlsx')
DF_consump['consumtion_date'] = pd.to_datetime(DF_consump['consumtion_date'])

DF_consump = DF_consump[DF_consump['qty'] > 0]
DF_consump = DF_consump.dropna(subset=['qty', 'product_category', 'consumtion_date'])

# print(abc_analisys())
# print(xyz_analisys())
# print(season())

abc_by_year = xyz_analisys()
for abc_df in abc_by_year:
    print(abc_df)



