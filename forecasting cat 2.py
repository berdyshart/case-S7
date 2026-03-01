import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


def abc_analysis(df, category = 0):
    def classify_abc(percentage):
        if percentage <= 80:
            return 'A'
        elif percentage <= 95:
            return 'B'
        else:
            return 'C'


    df = df.copy()
    df = df[df['product_category'] == category]
    abc_df = df.groupby('product_id')['amount'].sum().reset_index()


    abc_df = abc_df.sort_values(by='amount', ascending=False)

    abc_df['cum_sum'] = abc_df['amount'].cumsum()
    abc_df['cum_perc'] = abc_df['cum_sum'] / abc_df['amount'].sum() * 100

    abc_df['class'] = abc_df['cum_perc'].apply(classify_abc)

    return abc_df.set_index('product_id')['class'].to_dict()


def lead_time_fun(df, id):
    df = df.copy()
    df = df[df['product_id'] == id]
    df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days
    return df['lead_time'].mean()


def lead_time_stats(df, id):
    df = df[df['product_id'] == id].copy()
    df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days

    lt_avg = df['lead_time'].mean()
    lt_std = df['lead_time'].std()

    # если std NaN (1 наблюдение) — ставим 0
    if np.isnan(lt_std):
        lt_std = 0

    return lt_avg, lt_std


def get_historical_lead_times(df, product_id):
    df = df[df['product_id'] == product_id].copy()
    df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days
    lt = df['lead_time'].dropna().values

    if len(lt) == 0:
        return np.array([30])  # fallback

    return lt


def calculate_rop_bootstrap_empirical(historical_demand,
                                      historical_lt,
                                      service_level=0.99,
                                      n_sim=8000):

    demand_samples = []

    # Берём высокий перцентиль LT, а не случайный
    lt_effective = int(np.percentile(historical_lt, 97))

    for _ in range(n_sim):

        lt = int(np.random.choice(historical_lt))

        if lt <= 0:
            lt = 1

        demand = np.sum(
            np.random.choice(historical_demand, size=lt, replace=True))

        demand_samples.append(demand)

    rop = np.percentile(demand_samples, service_level * 100)
    max_level = np.percentile(demand_samples, 99.5)

    return int(np.ceil(rop)), int(np.ceil(max_level))


# def calculate_rop(avg_size, avg_interval,
#                   lt_avg=81, lt_std=50,
#                   service_level=0.95):
#
#     # средний дневной спрос
#     mu_d = avg_size / avg_interval
#
#     # приближённая дисперсия дневного спроса
#     sigma_d = avg_size / np.sqrt(avg_interval)
#
#     # средний спрос за LT
#     mu_lt = mu_d * lt_avg
#
#     # дисперсия спроса за LT
#     var_lt = lt_avg * sigma_d**2 + (mu_d**2) * lt_std**2
#     sigma_lt = np.sqrt(var_lt)
#
#     # Z для нормального распределения
#     Z = norm.ppf(service_level)
#
#     rop = mu_lt + Z * sigma_lt
#
#     return np.ceil(rop), np.ceil(mu_lt)


def croston_method(ts, alpha=0.1):
    d = np.array(ts)
    n = len(d)
    # Массивы для хранения сглаженного размера (z) и интервала (p)
    z = np.zeros(n)
    p = np.zeros(n)

    # Инициализация первой ненулевой точкой
    first_idx = np.flatnonzero(d)[0]
    z[first_idx] = d[first_idx]
    p[first_idx] = first_idx + 1

    q = 1  # счетчик интервала
    for t in range(first_idx + 1, n):
        if d[t] > 0:
            z[t] = alpha * d[t] + (1 - alpha) * z[t - 1]
            p[t] = alpha * q + (1 - alpha) * p[t - 1]
            q = 1
        else:
            z[t] = z[t - 1]
            p[t] = p[t - 1]
            q += 1

    avg_size = z[-1]
    avg_interval = p[-1]
    return avg_size, avg_interval


def simulate_inventory(
        historical_demand,
        current_stock,
        Max,
        Min,
        historical_lt,
        reliability=1.0,
        horizon_days=365):

    days = pd.date_range(start='2025-09-01', periods=horizon_days)
    stock = current_stock

    pipeline = []  # (arrival_day, qty)
    orders_log = []
    stockouts = 0

    for day in days:

        # === 1. Приход поставок ===
        arrivals = [o for o in pipeline if o[0] == day]

        for arrival in arrivals:
            # реальная недопоставка
            received = np.random.binomial(arrival[1], reliability)
            stock += received

        pipeline = [o for o in pipeline if o[0] > day]

        # === 2. Спрос ===
        demand = np.random.choice(historical_demand)
        stock -= demand

        if stock < 0:
            stockouts += 1
            stock = 0

        # === 3. Inventory position ===
        on_order = sum(o[1] for o in pipeline)
        inventory_position = stock + on_order

        # === 4. Заказ ===
        if inventory_position <= Min:

            # bootstrap реального LT
            lead_time = int(np.random.choice(historical_lt))
            arrival_day = day + pd.Timedelta(days=lead_time)

            # корректировка на reliability
            order_qty = (Max - inventory_position) / reliability
            order_qty = int(np.ceil(order_qty))
            min_order_qty = 10
            order_qty = max(order_qty, min_order_qty)

            pipeline.append((arrival_day, order_qty))
            orders_log.append((day, order_qty))

    return orders_log, stock, stockouts


def prepare_price_model(price_series, max_annual_trend=0.5):
    df = price_series.sort_index().copy()
    df = df[df > 0]
    log_price = np.log(df)

    t = np.arange(len(log_price))
    b, a = np.polyfit(t, log_price, 1)

    # ограничиваем тренд: max_annual_trend ~ 50% в год
    # если t в днях, год ≈ 365 дней
    max_b = np.log(1 + max_annual_trend) / 365
    b = np.clip(b, -max_b, max_b)

    trend = a + b * t
    residuals = log_price.values - trend
    residuals = residuals - residuals.mean()

    return {
        "a": a,
        "b": b,
        "last_price": df.iloc[-1],
        "last_t": len(t),
        "residuals": residuals
    }


def simulate_future_prices(price_model, horizon_days, n_sim=1000):
    a = price_model["a"]
    b = price_model["b"]
    residuals = price_model["residuals"]
    last_t = price_model["last_t"]
    last_price = price_model["last_price"]

    last_log = np.log(last_price)

    simulations = []

    for _ in range(n_sim):
        sim_prices = []
        prev_log = last_log  # стартуем от последней фактической

        for h in range(1, horizon_days + 1):
            # инкремент тренда за шаг (чтобы не переоценивать рост/падение)
            trend_step = b
            boot_resid = np.random.choice(residuals)

            log_price_future = (a + b * (last_t + h)) + boot_resid
            price_future = np.exp(log_price_future)
            price_future = np.clip(price_future,
                                   0.5 * last_price,  # не падаем ниже половины
                                   3.0 * last_price)  # и не растём выше ×3 за год

            sim_prices.append(price_future)
            prev_log = log_price_future

        simulations.append(sim_prices)

    return np.array(simulations)


def main():
    df_global = pd.read_excel('C:\\Users\\berdy\\OneDrive\\Рабочий стол\\final_orders_train.csv.xlsx')
    df_global['order_date'] = pd.to_datetime(df_global['order_date'])
    df_global['delivery_date'] = pd.to_datetime(df_global['delivery_date'])
    df_global = df_global[df_global['qty'] > 0].dropna(subset=['qty', 'product_category'])
    df_consumption_global = pd.read_excel(
        'C:\\Users\\berdy\\OneDrive\\Рабочий стол\\final_consumtion_train.csv.xlsx')
    df_consumption_global['consumtion_date'] = pd.to_datetime(
        df_consumption_global['consumtion_date'])
    df_consumption_global = df_consumption_global[df_consumption_global['qty'] > 0].dropna(
        subset=['qty', 'product_category', 'consumtion_date'])
    product_stocks = {
    74: 80, 75: 280, 76: 4, 77: 7, 78: 6, 79: 2, 80: 0, 81: 293,
    82: 16, 83: 413, 84: 2, 85: 398, 86: 89, 87: 56, 88: 45, 89: 33,
    90: 29, 91: 2, 92: 2, 93: 212, 94: 144, 95: 34, 96: 38, 97: 87,
    98: 28, 99: 1, 100: 0, 101: 185, 102: 212, 103: 60, 104: 61,
    105: 45, 106: 117, 107: 0, 108: 0, 109: 196, 110: 278, 111: 59,
    112: 48, 113: 185, 114: 71, 115: 1, 116: 3, 117: 400, 118: 243,
    119: 51, 120: 39, 121: 67, 122: 24, 123: 0, 124: 682, 125: 275,
    126: 50, 127: 57
    }
    product_reliability = {
    74: 1, 75: 1, 76: 1, 77: 1, 78: 1, 79: 1, 80: 0.95, 81: 0.97,
    82: 1, 83: 0.91, 84: 1, 85: 0.98, 86: 0.96, 87: 0.99, 88: 1,
    89: 1, 90: 1, 91: 1, 92: 1, 93: 0.99, 94: 0.99, 95: 1, 96: 1,
    97: 1, 98: 1, 99: 1, 101: 0.95, 102: 1, 103: 1, 104: 1, 105: 1,
    106: 1, 109: 0.94, 110: 1, 111: 1, 112: 1, 113: 1, 114: 1, 115: 1,
    116: 1, 117: 0.96, 118: 1, 119: 1, 120: 1, 121: 1, 122: 1, 124: 0.95,
    125: 0.95, 126: 1, 127: 1
    }
    ids = [74, 75, 79, 80, 81, 82, 83, 85, 86, 87, 88, 93, 94, 95, 96, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127]
    abc = abc_analysis(df_global, category = 2)
    service_by_class = {'A': 0.99, 'B': 0.99, 'C': 0.99}

    total_costs_by_cat0 = 0
    for id in ids:
        print('---', id, '---')

        df_consumption = df_consumption_global[
            df_consumption_global['product_id'] == id
            ]
        df = df_global[df_global['product_id'] == id]

        if len(df) < 2 or df_consumption.empty:
            print(f"Пропускаем ID {id}: недостаточно данных")
            continue

        # ---------------- ЦЕНА ----------------
        price_series = df.set_index('order_date')['price']
        price_model = prepare_price_model(price_series)

        # ---------------- СПРОС ----------------
        ts = df_consumption.groupby('consumtion_date')['qty'].sum()
        ts.index = pd.to_datetime(ts.index)
        ts = ts.sort_index()

        # приводим к ежедневной частоте
        ts = ts.resample('D').sum()

        # расширяем до общего календаря
        start = ts.index.min()
        end = ts.index.max()
        full_range = pd.date_range(start, end, freq='D')
        ts = ts.reindex(full_range, fill_value=0)

        # ---------------- ПОСЛЕДНИЙ ГОД ----------------
        last_date = ts.index.max()
        start_365 = last_date - pd.Timedelta(days=364)

        recent_ts = ts.loc[start_365:last_date]
        lambda_fact = recent_ts.sum() / 365

        target_sl = service_by_class[abc.get(id, 'B')]
        lt_avg, lt_std = lead_time_stats(df_global, id)
        historical_lt = get_historical_lead_times(df_global, id)
        # ==================================================
        # ================= RARE SKU =======================
        # ==================================================
        cap = np.percentile(ts.values, 99)
        ts_capped = np.minimum(ts.values, cap)
        historical_demand = ts_capped
        print("max daily demand", ts.max())
        print("top 5 daily demand", np.sort(ts.values)[-5:])
        if ts.sum() < 100:
            print('rare')

            # средний дневной спрос
            lambda_daily = ts.sum() / len(ts)

            # моделируем LT как случайную величину
            lt_samples = np.random.choice(historical_lt, size=5000)

            # спрос за LT ~ Poisson(lambda * LT)
            demand_samples = np.random.poisson(lambda_daily * lt_samples)

            rop = np.percentile(demand_samples, 99.5)
            max_level = np.percentile(demand_samples, 99.9)


            Min = int(np.ceil(rop))
            Max = int(np.ceil(max_level))

        # ==================================================
        # ================= NORMAL SKU =====================
        # ==================================================
        else:
            print('norm')
            ts_last_2_years = ts[-730:]

            avg_size, avg_interval = croston_method(ts_last_2_years)
            lambda_croston = avg_size / avg_interval
            lambda_fact = recent_ts.sum() / 365

            # сглаживание через вес
            w = 0.5
            lambda_used = w * lambda_croston + (1 - w) * lambda_fact

            historical_lt = get_historical_lead_times(df_global, id)

            rop, max_level = calculate_rop_bootstrap_empirical(
                historical_demand=ts_last_2_years.values,
                historical_lt=historical_lt,
                service_level=target_sl
            )

            Min = rop
            Max = max_level

        # защита от слишком маленького диапазона
        if Max <= Min:
            Max = int(Min * 1.5)

        # защита от нуля
        Min = max(Min, 5)
        Max = max(Max, Min + 5)

        print('rop', rop)
        print('min', Min)
        print('max', Max)


        current_stock = product_stocks.get(id, 0)
        reliability = product_reliability.get(id, 0)

        n_sim = 500
        total_costs = []
        price_avg = []
        stockouts_final = []
        price_sims = simulate_future_prices(price_model, 365, n_sim=n_sim)
        for i in range(n_sim):
            orders, stock, stockouts = simulate_inventory(
                historical_demand=historical_demand,
                current_stock=current_stock,
                Max=Max,
                Min=Min,
                historical_lt=historical_lt,
                reliability=reliability
            )
            total = 0
            for order_date, qty in orders:
                day_index = (order_date - pd.Timestamp("2025-09-01")).days
                if day_index >= 365:
                    continue
                price = price_sims[i, day_index]
                total += qty * price
                price_avg.append(price)
            total_costs.append(total)
            stockouts_final.append(stockouts)
            if i == 0:
                print('orders', orders)
        if len(price_avg) == 0:
            avg_price = np.nan
        else:
            avg_price = np.mean(price_avg)
        print(historical_lt.mean())
        print(historical_lt.max())

        print('price_avg', avg_price)
        print('stockouts', np.mean(stockouts_final))
        print("Mean cost:", np.mean(total_costs))
        print("95% budget:", np.percentile(total_costs, 95))
        total_costs_by_cat0 += np.mean(total_costs)
    print('total_costs_by_cat0', total_costs_by_cat0)

main()
