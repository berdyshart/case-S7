import pandas as pd
import matplotlib.pyplot as plt


def volatility_of_price_fun(df):
    df = df.copy()
    price_volatility = df.groupby('product_category')['price'].std() / \
                       df.groupby('product_category')['price'].mean()
    return price_volatility


def lead_time_fun(df):
    df = df.copy()
    df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days
    avg_lead_time = df.groupby('product_category')['lead_time'].mean()
    return avg_lead_time


def fillrate_fun(df):
    df = df.copy()
    delivery_relatability = df.groupby('product_category')[
                                'valid_delivered_qty'].mean() / \
                            df.groupby('product_category')['qty'].mean()
    return delivery_relatability


def seasonality_fun(df):
    df = df.copy()
    df.set_index('order_date', inplace=True)
    seasonality = df.groupby('product_category')['price'].resample(
        'ME').count().unstack(level=0)

    seasonality.plot(figsize=(12, 6), marker='o')
    plt.title('Сезонность закупок по категориям (объем закупок по месяцам)')
    plt.xlabel('Дата')
    plt.ylabel('Объем закупок')
    plt.grid(True)
    plt.legend(title='Категории')
    plt.show()


def consumption_fun(df_consumption):
    df = df_consumption.copy()
    df.set_index('consumtion_date', inplace=True)
    consumption = df.groupby('product_category')['qty'].resample(
        'ME').count().unstack(level=0)

    consumption.plot(figsize=(12, 6), marker='o')
    plt.title('Сезонность потребления по категориям (qty по месяцам)')
    plt.xlabel('Дата')
    plt.ylabel('Количество потребленное')
    plt.grid(True)
    plt.legend(title='Категории')
    plt.show()


def abc_analysis_by_year(df):
    result = []

    def classify_abc(prev_percentage):
        if prev_percentage < 80:
            return 'A'
        elif prev_percentage < 95:
            return 'B'
        else:
            return 'C'

    df['year'] = df['order_date'].dt.year
    abc_df_temp = df.groupby(['year', 'product_category'])[
        'amount'].sum().reset_index()
    years = sorted(df['year'].unique())

    for year in years:
        abc_df = abc_df_temp[abc_df_temp['year'] == year]
        abc_df = abc_df.sort_values(by='amount', ascending=False)

        abc_df['cum_sum'] = abc_df['amount'].cumsum()
        abc_df['cum_perc'] = abc_df['cum_sum'] / abc_df['amount'].sum() * 100

        abc_df['prev_cum_perc'] = abc_df['cum_perc'].shift(1).fillna(0)
        abc_df['class'] = abc_df['prev_cum_perc'].apply(classify_abc)

        abc_df = abc_df.drop(columns=['prev_cum_perc'])
        result.append(abc_df)

    return result


def xyz_analisis_by_year(df_consumption):
    df_consumption = df_consumption.copy()
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
                                     labels=['X', 'Y', 'Z']),
        include_lowest = True
        xyz_df['year'] = year
        result.append(xyz_df)

    return result


def main():
    # импорт данных
    df = pd.read_excel(
        'C:\\Users\\berdy\\OneDrive\\Рабочий стол\\final_orders_train.csv.xlsx')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df = df[df['qty'] > 0].dropna(subset=['qty', 'product_category'])
    df_consumption = pd.read_excel(
        'C:\\Users\\berdy\\OneDrive\\Рабочий стол\\final_consumtion_train.csv.xlsx')
    df_consumption['consumtion_date'] = pd.to_datetime(
        df_consumption['consumtion_date'])
    df_consumption = df_consumption[df_consumption['qty'] > 0].dropna(
        subset=['qty', 'product_category', 'consumtion_date'])

    abc = abc_analysis_by_year(df)
    for i in abc:
        print(i)
main()