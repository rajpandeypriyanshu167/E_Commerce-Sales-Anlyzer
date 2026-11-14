import matplotlib.pyplot as plt
import pandas as pd

# ---------- 1. Load & clean ----------
df = pd.read_csv("ecommerce_dataset.csv")
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df = df.drop_duplicates(subset="order_id")
df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()

print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------- 2. KPIs ----------
status = df["order_status"].str.lower()
bad_orders = status.str.contains("cancel|return", na=False)

kpis = {
    "Total Revenue": df["total_amount"].sum(),
    "Total Orders": df["order_id"].nunique(),
    "Unique Customers": df["customer_id"].nunique(),
    "Avg Order Value": df["total_amount"].mean(),
    "Avg Rating": df["rating"].mean(),
    "Avg Shipping Days": df["shipping_days"].mean(),
    "Avg Discount %": df["discount_pct"].mean(),
    "Cancel/Return Rate %": bad_orders.mean() * 100,
}

print("\n========== KPIs ==========")
for name, value in kpis.items():
    print(f"{name:<22}: {value:,.2f}")

# ---------- 3. Aggregations ----------
monthly_rev = df.groupby("month")["total_amount"].sum()
cat_rev = df.groupby("category")["total_amount"].sum().sort_values()
status_counts = df["order_status"].value_counts()
pay_counts = df["payment_method"].value_counts()
top_states = df.groupby("state")["total_amount"].sum().nlargest(10).sort_values()
device_rating = df.groupby("device")["rating"].mean()

# ---------- 4. Dashboard (6 charts + KPI banner) ----------
fig = plt.figure(figsize=(18, 12))
fig.suptitle("E-commerce Sales Dashboard", fontsize=20, fontweight="bold")

# KPI banner
ax_kpi = fig.add_axes([0.03, 0.88, 0.94, 0.06])
ax_kpi.axis("off")
items = list(kpis.items())
for i, (name, value) in enumerate(items):
    x = (i + 0.5) / len(items)
    txt = f"{value:,.0f}" if value > 100 else f"{value:.2f}"
    ax_kpi.text(x, 0.65, txt, ha="center", fontsize=15, fontweight="bold", color="#1f4e79")
    ax_kpi.text(x, 0.15, name, ha="center", fontsize=9, color="gray")

gs = fig.add_gridspec(2, 3, top=0.82, bottom=0.07, hspace=0.35, wspace=0.3)

# Chart 1: Monthly revenue trend (line)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(monthly_rev.index, monthly_rev.values, marker="o", color="#1f77b4")
ax1.set_title("Monthly Revenue Trend")
ax1.tick_params(axis="x", rotation=45)
ax1.grid(alpha=0.3)

# Chart 2: Revenue by category (horizontal bar)
ax2 = fig.add_subplot(gs[0, 1])
ax2.barh(cat_rev.index, cat_rev.values, color="#2ca02c")
ax2.set_title("Revenue by Category")

# Chart 3: Order status (pie)
ax3 = fig.add_subplot(gs[0, 2])
ax3.pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%", startangle=90)
ax3.set_title("Order Status Share")

# Chart 4: Payment methods (bar)
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(pay_counts.index, pay_counts.values, color="#ff7f0e")
ax4.set_title("Orders by Payment Method")
ax4.tick_params(axis="x", rotation=30)

# Chart 5: Customer age distribution (histogram)
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df["customer_age"].dropna(), bins=15, color="#9467bd", edgecolor="white")
ax5.axvline(df["customer_age"].mean(), color="red", linestyle="--", label="Mean")
ax5.set_title("Customer Age Distribution")
ax5.legend()

# Chart 6: Top 10 states by revenue (horizontal bar)
ax6 = fig.add_subplot(gs[1, 2])
ax6.barh(top_states.index, top_states.values, color="#d62728")
ax6.set_title("Top 10 States by Revenue")

plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------- 5. Extra: correlation (numpy) ----------
num_cols = ["customer_age", "unit_price", "quantity", "discount_pct",
            "total_amount", "shipping_days", "rating"]
print("\nCorrelation matrix:\n", df[num_cols].corr().round(2))
