import pandas as pd
import numpy as np
from scipy.stats import norm


def simulate_inventory(
        ts,
        current_stock,
        lead_time,
        horizon=365,
        service_quantile=0.97,
        alpha=0.1,
        review_period=1):

    sims_lt = monte_carlo_demand(ts, horizon=lead_time, n_sim=3000, alpha=alpha)

    if sims_lt is None:
        return []

    z = norm.ppf(service_quantile)

    mean_daily = croston_sba(ts)[0]
    std_daily = np.std(ts)

    mean_LT = mean_daily * lead_time
    std_LT = std_daily * np.sqrt(lead_time)

    ROP = mean_LT + z * std_LT

    mean_daily = croston_sba(ts, alpha)[0]
    S = ROP + mean_daily * 30   # держим месяц сверху

    stock = current_stock
    orders = []
    pipeline = []

    positive_values = ts[ts > 0]
    p = min(1, 1 / croston_sba(ts, alpha)[2])

    for day in range(horizon):

        # приход поставок
        arrivals = [o for o in pipeline if o[0] == day]
        for a in arrivals:
            stock += a[1]
        pipeline = [o for o in pipeline if o[0] > day]

        # генерируем спрос
        if np.random.rand() < p:
            demand = np.random.choice(positive_values)
        else:
            demand = 0

        stock -= demand

        # проверка точки заказа
        if stock + sum([o[1] for o in pipeline]) <= ROP:
            order_qty = S - stock - sum([o[1] for o in pipeline])
            arrival_day = day + lead_time

            pipeline.append((arrival_day, order_qty))

            orders.append((day,order_qty))

    return orders


def croston_sba(ts, alpha=0.1):
    ts = np.array(ts)
    demand = ts.copy()

    non_zero_indices = np.where(demand > 0)[0]

    if len(non_zero_indices) == 0:
        return 0, 0, 0

    first = non_zero_indices[0]
    q_hat = demand[first]

    if len(non_zero_indices) > 1:
        a_hat = non_zero_indices[1] - non_zero_indices[0]
    else:
        a_hat = len(demand)

    interval = 1

    for t in range(first + 1, len(demand)):
        if demand[t] > 0:
            q_hat = q_hat + alpha * (demand[t] - q_hat)
            a_hat = a_hat + alpha * (interval - a_hat)
            interval = 1
        else:
            interval += 1

    croston_forecast = q_hat / a_hat if a_hat > 0 else 0
    sba_forecast = croston_forecast * (1 - alpha / 2)

    return sba_forecast, q_hat, a_hat


def monte_carlo_demand(ts, horizon=365, n_sim=10000, alpha=0.1):

    mean_demand, q_hat, a_hat = croston_sba(ts, alpha)

    if a_hat <= 0:
        return None

    p = min(1, 1 / a_hat)

    positive_values = ts[ts > 0]
    if len(positive_values) == 0:
        return None

    simulations = np.zeros(n_sim)

    for i in range(n_sim):
        demand_path = np.random.rand(horizon) < p
        sizes = np.random.choice(positive_values, size=horizon, replace=True)
        simulations[i] = np.sum(demand_path * sizes)

    return simulations


def main():
    ids = [74, 75, 79, 80, 81, 82, 83, 85, 86, 87, 88, 93, 94,
           95, 96, 100, 101, 102, 103, 104, 105, 107,
           108, 109, 110, 111, 112, 117, 118, 119,
           120, 122, 123, 124, 125, 126, 127]

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
    # Загружаем данные
    cons = pd.read_excel(
            'C:\\Users\\berdy\\OneDrive\\Рабочий стол\\final_consumtion_train.csv.xlsx')

    # Фильтр категории 2
    cons = cons[cons["product_category"] == 2].copy()

    # Дата
    cons["consumption_date"] = pd.to_datetime(cons["consumption_date"])

    # Агрегируем до дневного спроса по SKU
    daily = (
        cons
        .groupby(["product_id", "consumption_date"])["qty"]
        .sum()
        .reset_index()
    )

    # Создаём полный календарь по каждому SKU
    def create_full_series(df):
        results = {}
        for pid in df["product_id"].unique():
            tmp = df[df["product_id"] == pid].copy()
            full_idx = pd.date_range(tmp["consumption_date"].min(),
                                     tmp["consumption_date"].max(),
                                     freq="D")
            tmp = tmp.set_index("consumption_date").reindex(full_idx)
            tmp["qty"] = tmp["qty"].fillna(0)
            results[pid] = tmp["qty"]
        return results

    series_dict = create_full_series(daily)

    results = {}

    for pid in ids:

        if pid not in series_dict:
            results[pid] = {
                "mean_annual_demand": 0,
                "p95_annual_demand": 0,
                "p99_annual_demand": 0
            }
            continue

        ts = series_dict[pid]

        sims = monte_carlo_demand(ts.values, horizon=365, n_sim=5000)

        if sims is None:
            results[pid] = {
                "mean_annual_demand": 0,
                "p95_annual_demand": 0,
                "p99_annual_demand": 0
            }
            continue

        results[pid] = {
            "mean_annual_demand": sims.mean(),
            "p95_annual_demand": np.percentile(sims, 95),
            "p99_annual_demand": np.percentile(sims, 99)
        }

    forecast_df = pd.DataFrame(results).T
    forecast_df.head()

    lead_time = 60

    rop_results = {}

    for pid, ts in series_dict.items():
        sims = monte_carlo_demand(ts.values, horizon=lead_time, n_sim=5000)

        if sims is None:
            continue

        rop_results[pid] = {
            "mean_LT_demand": sims.mean(),
            "p97_LT_demand": np.percentile(sims, 97),
            "p99_LT_demand": np.percentile(sims, 99)
        }

    rop_df = pd.DataFrame(rop_results).T
    rop_df.head()




    for pid in ids:
        print(pid)
        if pid not in series_dict:
            continue

        ts = series_dict[pid].values
        current_stock = product_stocks.get(pid, 0)

        orders = simulate_inventory(
            ts,
            current_stock=current_stock,
            lead_time=60,
            horizon=365,
            service_quantile=0.97
        )

        print(orders)
        print(sum([o[1] for o in orders]))


main()


