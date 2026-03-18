import datetime
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
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
    # which tends to happen around the start of the data
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
        (pl.col(return_col) * pl.col(flag_col).shift(-1)).alias(f"return earned: {flag_col} (shifted)")
    ).with_columns(
        ((pl.col(f"return earned: {flag_col} (shifted)")/base + 1).cum_prod().alias(f"cumret: {flag_col}"))
    )
    
def build_ratio_table(df: pl.DataFrame, front_pd: str, back_pd: str) -> pl.DataFrame:
    df = df.select([
        pl.col("Trade Date"),
        pl.col("BATS:VIXY - Return"),
        pl.col(f"t_{front_pd}"),
        pl.col(f"t_{back_pd}")
    ])
    df = df.with_columns([
        (pl.col(f"t_{front_pd}") / pl.col(f"t_{back_pd}")).alias(f"{front_pd}/{back_pd} Ratio"),
        ((pl.col(f"t_{front_pd}") / pl.col(f"t_{back_pd}")) > 1).alias(f"{front_pd}/{back_pd} inversion"),
        pl.col("BATS:VIXY - Return").shift(-1).alias("BATS:VIXY - Return (Shifted)")
    ])
    
    # doing some cleanup here:
    #   dropping nulls from .shift(1)
    #   have some data issues on the tail end because our interpolation func uses the last value if it doesn't
    #   have a back data point to interpolate from. this leads to our ratio becoming exactly 1.0 for the last few obs
    df = df.drop_nulls()
    df = df.filter((pl.col(f"{front_pd}/{back_pd} Ratio") != 1.0))
    
    return df
    
def build_strategy_table(df: pl.DataFrame, front_pd: str, back_pd: str) -> pl.DataFrame:
    df = df.select([
        pl.col("Trade Date"),
        pl.col("BATS:VIXY - Return"),
        pl.col(f"t_{front_pd}"),
        pl.col(f"t_{back_pd}")
    ])
    df = df.with_columns([
        (pl.col(f"t_{front_pd}") / pl.col(f"t_{back_pd}")).alias(f"{front_pd}/{back_pd} Ratio"),
        ((pl.col(f"t_{front_pd}") / pl.col(f"t_{back_pd}")) > 1).alias(f"{front_pd}/{back_pd} inversion"),
        (((pl.col(f"t_{front_pd}") / pl.col(f"t_{back_pd}")) > 1).shift(1) * pl.col("BATS:VIXY - Return")).alias("Return Earned")
    ])
    
    df = df.drop_nulls()
    df = df.filter((pl.col(f"{front_pd}/{back_pd} Ratio") != 1.0))
    
    return df 
    
def bucket_on_ratio(df: pl.DataFrame, bucket_ratio: str, fwd: bool = True, n_buckets: int = 4) -> pl.DataFrame:
    bucket = df.with_columns(pl.col(f"{bucket_ratio} Ratio").qcut(n_buckets).alias(f"Ratio Bucket {bucket_ratio}"))
    out = (
        bucket.group_by(f"Ratio Bucket {bucket_ratio}")
            .agg([
                pl.len().alias("n"),
                pl.col(f"{bucket_ratio} Ratio").mean().alias("Average Ratio"),
                pl.col(f"BATS:VIXY - Return{' (Shifted)' if fwd else ''}").mean().alias("Mean Return"),
                pl.col(f"BATS:VIXY - Return{' (Shifted)' if fwd else ''}").median().alias("Median Return"),
                pl.col(f"BATS:VIXY - Return{' (Shifted)' if fwd else ''}").std().alias("Std Dev"),                
            ]).sort(by=f"Ratio Bucket {bucket_ratio}")
    )
    
    return out

def bucket_on_ratio_inverted(df: pl.DataFrame, bucket_ratio: str, fwd: bool = True, n_buckets: int = 4) -> pl.DataFrame:
    inverted = df.filter(pl.col(f"{bucket_ratio} Ratio") > 1)
    bucket = inverted.with_columns(pl.col(f"{bucket_ratio} Ratio").qcut(n_buckets).alias(f"Ratio Bucket {bucket_ratio}"))
    out = (
        bucket.group_by(f"Ratio Bucket {bucket_ratio}")
            .agg([
                pl.len().alias("n"),
                pl.col(f"{bucket_ratio} Ratio").mean().alias("Average Ratio"),
                pl.col(f"BATS:VIXY - Return{' (Shifted)' if fwd else ''}").mean().alias("Mean Return"),
                pl.col(f"BATS:VIXY - Return{' (Shifted)' if fwd else ''}").median().alias("Median Return"),
                pl.col(f"BATS:VIXY - Return{' (Shifted)' if fwd else ''}").std().alias("Std Dev"),
                (pl.col(f"BATS:VIXY - Return{' (Shifted)' if fwd else ''}") > 0).mean().alias("Hit Rate")
            ]).sort(by=f"Ratio Bucket {bucket_ratio}")
    )
    
    return out

def draw_wealth(df: pl.DataFrame, ratio: str, base: int | None = 100) -> pl.DataFrame:
    df = df.with_columns(
        ((pl.col("Return Earned") / base + 1).cum_prod()).alias(f"{ratio} Cumulative Return")
    )
    
    fig, ax = plt.subplots()
    plt.plot(df["Trade Date"], df[f"{ratio} Cumulative Return"])
    plt.show()
    
    return df