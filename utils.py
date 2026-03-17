import datetime
import polars as pl
import numpy as np
from dateutil.relativedelta import relativedelta

# takes in a pl.Date in the format <month><year>
# vx futures expire 30 days prior to the third friday of the next month
# so a vx Jan 2026 contract will expire 30 days prior to 20th Feb 2026
def find_contract_expiry(c_date: datetime.date) -> datetime.date:
    next = c_date + relativedelta(months=1)
    exp = datetime.date(year=next.year, month=next.month, day=15) # day set to 15 to find the third friday
    while exp.weekday() != 4:
        exp += datetime.timedelta(days=1)
    exp -= datetime.timedelta(days=30)
    return exp

def interp_price(x: list[float], y: list[float], target: float):
    for xi, yi in zip(x, y):
        if xi == target:
            return yi
        
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        
        if x0 <= target <= x1:
            return y0
        return y0 + (target - x0) * (y1 - y0) / (x1 - x0)
    
    return None

TARGET_TENORS = [7, 30, 60, 90, 120]
def interp_targets(group: pl.DataFrame):
    dte = group["Days to Expiry"].cast(pl.Float64).to_list()
    prices = group["Settle"].cast(pl.Float64).to_list()
    dates = group["Trade Date"].cast(pl.Date)[0]

    row = {"Trade Date": dates}
    for target in TARGET_TENORS:
        row[f"t_{int(target)}"] = interp_price(dte, prices, target)
        
    # we have to manually define the scheme in case there are nulls in our output,
    # which tends to happen around the start of the data where we don't have full data
    return pl.DataFrame([row], schema={
        "Trade Date": group.schema["Trade Date"],
        "t_7": pl.Float64,
        "t_30": pl.Float64,
        "t_60": pl.Float64,
        "t_90": pl.Float64,
        "t_120": pl.Float64
    })
    
def add_cumret(
    df: pl.DataFrame,
    flag_col: str,
    return_col: str,
    base: int | None = 100
):
    if not base:
        base = 1
    
    return df.with_columns(
        (pl.col(return_col) * pl.col(flag_col).shift(1)).alias(f"return earned: {flag_col} (shifted)")
    ).with_columns(
        ((pl.col(f"return earned: {flag_col} (shifted)")/base + 1).cum_prod().alias(f"cumret: {flag_col}"))
    )
    
    