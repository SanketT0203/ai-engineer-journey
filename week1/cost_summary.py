import os 
import pandas as pd

df=pd.read_csv("usage_log.csv", parse_dates=["timestamp"])
print("=== Totals by provider/model ===")

print(
    df.groupby(["provider", "model"])
    .agg(calls=("cost_eur", "count"),
         total_input_tokens=("input_tokens", "sum"),
         total_output_tokens=("output_tokens", "sum"),
         total_cost_eur=("cost_eur", "sum"))
    .round(4)
)

print(f"\n=== Grand total spent: €{df['cost_eur'].sum():.4f} across {len(df)} calls ===")
 
print("\n=== Spend by day ===")
daily = df.set_index("timestamp").resample("D")["cost_eur"].sum()
print(daily[daily > 0].round(4))