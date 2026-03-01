import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


def lead_time_fun(df):
    df = df.copy()
    df = df[df['product_id'] == 14]
    df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days
    return df['lead_time'].mean()


def calculate_rop(avg_size, avg_interval,
                  lt_avg=81, lt_std=50,
                  service_level=0.95):

    # средний дневной спрос
    mu_d = avg_size / avg_interval

    # приближённая дисперсия дневного спроса
    sigma_d = avg_size / np.sqrt(avg_interval)

    # средний спрос за LT
    mu_lt = mu_d * lt_avg

    # дисперсия спроса за LT
    var_lt = lt_avg * sigma_d**2 + (mu_d**2) * lt_std**2
    sigma_lt = np.sqrt(var_lt)

    # Z для нормального распределения
    Z = norm.ppf(service_level)

    rop = mu_lt + Z * sigma_lt

    return np.ceil(rop), np.ceil(mu_lt)


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


def simulate_inventory(avg_size, avg_interval, current_stock, Max, Min,
                       lt_avg=81, lt_std=50, reliability = 1.0):
    days = pd.date_range(start='2025-09-01', end='2026-08-31')
    stock = current_stock
    prob = 1 / avg_interval

    pipeline = []  # заказы в пути (день_прибытия, количество)
    orders_log = []

    for day in days:
        # 1. Проверяем приход поставок
        arrivals = [o for o in pipeline if o[0] == day]
        for arrival in arrivals:
            good_qty = np.random.binomial(arrival[1], reliability)
            stock += good_qty

        pipeline = [o for o in pipeline if o[0] > day]

        # 2. Генерируем спрос
        if np.random.rand() < prob:
            demand = max(0, np.random.normal(avg_size, avg_size * 0.2))
        else:
            demand = 0

        stock -= np.ceil(demand)

        # 3. Проверяем ROP
        on_order = sum([o[1] for o in pipeline])
        inventory_position = on_order + stock

        if inventory_position <= Min:
            # print(day, inventory_position)
            lead_time = max(1, int(np.random.normal(lt_avg, lt_std)))
            arrival_day = day + pd.Timedelta(days=lead_time)

            order_qty = Max - inventory_position
            pipeline.append((arrival_day, int(np.ceil(order_qty))))
            orders_log.append((day, int(np.ceil(order_qty))))
        if stock < 0:
            stock = 0
            # print(day, "stockout!!")

    return orders_log, stock


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

            log_price_future = prev_log + trend_step + boot_resid
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
    df_consumption_global['consumption_date'] = pd.to_datetime(
        df_consumption_global['consumption_date'])
    df_consumption_global = df_consumption_global[df_consumption_global['qty'] > 0].dropna(
        subset=['qty', 'product_category', 'consumption_date'])
    product_stocks = {
        7: 2,
        8: 2,
        11: 227 + 21,
        12: 12,
        13: 141,
        14: 122 + 36,
        16: 244,
        17: 35,
        18: 36 + 81,
        19: 95 + 33,
        20: 0,
        27: 83,
        28: 17,
        29: 65
    }
    product_reliability = {
        7: 1.0,
        8: 0.0,
        11: 1.0,
        12: 1.0,
        13: 0.98,
        14: 0.97,
        16: 1.0,
        17: 0.98,
        18: 0.97,
        19: 1.0,
        20: 1.0,
        27: 0.9,
        28: 1.0,
        29: 0.96
    }
    ids = [11, 12, 13, 14, 16, 17, 19]
    abc = abc_analysis(df_global)
    service_by_class = {'A': 0.98, 'B': 0.95, 'C': 0.9}

    total_costs_by_cat0 = 0
    final_orders = []
    for id in ids:
        print('---', id, '---')
        df_consumption = df_consumption_global[df_consumption_global['product_id'] == id]
        df = df_global[df_global['product_id'] == id]
        if len(df) < 2 or df_consumption.empty:
            print(f"Пропускаем ID {id}: недостаточно данных для модели (нужно минимум 2 цены)")
            continue
        price_series = df.set_index('order_date')['price']
        price_model = prepare_price_model(price_series)
        ts = df_consumption.groupby('consumption_date')['qty'].sum()
        full_range = pd.date_range(start=ts.index.min(), end=ts.index.max(),
                                   freq='D')
        ts = ts.reindex(full_range, fill_value=0)
        target_sl = 0.98
        total_demand = ts.sum()
        days = len(ts)
        lambda_fact = total_demand / days if days > 0 else 0

        if ts.sum() < 100:
            print("Low turnover SKU → simple annual planning")

            total_demand = ts.sum()
            days = len(ts)

            lambda_fact = total_demand / 365 / 1.2 if days > 0 else 0

            # прогноз на год
            annual_forecast = lambda_fact * 365

            # небольшой страховой запас 10%
            annual_forecast *= 1.1

            annual_qty = int(np.ceil(annual_forecast))

            # средняя цена
            avg_price = df['price'].mean()

            total_budget = annual_qty * avg_price

            print("Annual demand forecast:", annual_qty)
            print("Planned annual закупка:", annual_qty)
            print("Estimated budget:", total_budget)

            print(total_budget)
            total_costs_by_cat0 += total_budget
            order_date = pd.Timestamp("2025-09-01")

            final_orders.append({
                "order_date": order_date,
                "product_category": 0,
                "product_id": id,
                "order_qty": annual_qty,
                "amount": annual_qty * avg_price
            })
            continue

        avg_size, avg_interval = croston_method(ts)
        lambda_croston = avg_size / avg_interval  # ед./день из Кростона

        # не даём модели «летать» выше факта больше чем на 30 %
        lambda_used = min(lambda_croston, lambda_fact * 1.3)

        # пересчитываем avg_size под новый темп (интервал оставляем)
        avg_size = lambda_used * avg_interval

        rop, mean_lt_demand = calculate_rop(avg_size, avg_interval, service_level=target_sl)
        # ограничиваем ROP по разумному покрытию
        coverage_months = 2  # максимум 2 месяца спроса в точке заказа
        max_rop_by_coverage = lambda_used * 30 * coverage_months  # спрос за 2 мес

        rop = min(rop, max_rop_by_coverage)

        mean_lt_demand = lambda_used * lead_time_fun(df_global)
        Max = rop
        Min = rop - mean_lt_demand / 2

        current_stock = product_stocks.get(id, 0)
        reliability = product_reliability.get(id, 0)
        print("Min:", Min, "Max:", Max, "stock:", current_stock)
        print("avg_size:", avg_size, "avg_interval:", avg_interval)

        n_sim = 500
        total_costs = []
        price_avg = []
        for s in range(n_sim):
            orders, stock = simulate_inventory(avg_size=avg_size,
                                               avg_interval=avg_interval,
                                               current_stock=current_stock,
                                               Max=Max, Min=Min,
                                               reliability=reliability)

            price_sims = simulate_future_prices(price_model, 365, n_sim=1)
            total = 0
            for order_date, qty in orders:
                day_index = (order_date - pd.Timestamp("2025-09-01")).days
                price = price_sims[0, day_index]
                total += qty * price
                price_avg.append(price)
                if s == 0:
                    final_orders.append({
                        "order_date": order_date,
                        "product_category": 0,
                        "product_id": id,
                        "order_qty": qty,
                        "amount": qty * price
                    })
            total_costs.append(total)

        print(np.mean(price_avg))
        print("Mean cost:", np.mean(total_costs))
        print("95% budget:", np.percentile(total_costs, 95))
        total_costs_by_cat0 += np.mean(total_costs)
    print('total_costs_by_cat0', total_costs_by_cat0)
    final_df = pd.DataFrame(final_orders)

    if not final_df.empty:
        final_df = final_df.sort_values("order_date")

        print("\n=== FINAL PROCUREMENT PLAN ===")
        print(final_df)

        final_df.to_excel("final_procurement_plan_cat0_final.xlsx", index=False)
main()
