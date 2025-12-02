import os

import altair as alt
import pandas as pd
from vega_datasets import data

# Allow large data
alt.data_transformers.disable_max_rows()



# 1. Data loading
def load_data():
    """
    Load cleaned CSVs from the Data/ folder.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df_map = pd.read_csv(os.path.join(current_dir, "Data", "clean_map.csv"))
    df_age = pd.read_csv(os.path.join(current_dir, "Data", "clean_heatmap.csv"))
    df_income = pd.read_csv(os.path.join(current_dir, "Data", "clean_income.csv"))
    return df_map, df_age, df_income



# 2. Helper: weighted mean using Sample_Size
def weighted_mean(group, value_col="Data_Value", weight_col="Sample_Size"):
    """
    Compute weighted mean of value_col using weight_col as weights.
    """
    w = group[weight_col]
    x = group[value_col]
    return (x * w).sum() / w.sum()


# 3. Viz 1: Age (heatmap)
def make_age_heatmap(df_age: pd.DataFrame) -> alt.Chart:
    """
    Make heatmap of obesity by age group over time (national average).
    df_age is expected to have:
      - YearStart
      - Stratification1 (age group)
      - Data_Value
      - Sample_Size
    """

    df = df_age.copy()
    df["Sample_Size"] = pd.to_numeric(df["Sample_Size"], errors="coerce")
    df = df.dropna(subset=["Sample_Size", "Data_Value"])

    df_nat = (
        df.groupby(["YearStart", "Stratification1"])
        .apply(weighted_mean)
        .reset_index(name="obesity_rate")
        .rename(columns={"YearStart": "year", "Stratification1": "age_group"})
    )

    age_order = [
        "18 - 24",
        "25 - 34",
        "35 - 44",
        "45 - 54",
        "55 - 64",
        "65 or older",
    ]

    chart = (
        alt.Chart(df_nat)
        .mark_rect()
        .encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("age_group:N", title="Age group", sort=age_order),
            color=alt.Color(
                "obesity_rate:Q",
                title="Obesity rate (%)",
                scale=alt.Scale(scheme="yelloworangebrown"),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("age_group:N", title="Age group"),
                alt.Tooltip(
                    "obesity_rate:Q", title="Obesity (%)", format=".1f"
                ),
            ],
        )
        .properties(
            width=500,
            height=320,
            title="Adult obesity by age group over time (US average)",
        )
    )

    return chart



# 4. Viz 2: National trend + map (linked by year)
def make_trend_and_map(df_map_raw: pd.DataFrame) -> alt.VConcatChart:
    """
    Make a vertical layout:
      - top: national average trend over time
      - bottom: US map colored by obesity rate for the hovered year
    df_map_raw is expected to have:
      - YearStart
      - LocationAbbr
      - LocationDesc
      - Data_Value
    """

    df_map = df_map_raw.copy()

    # Selection by year (classic selection for better compatibility)
    hover_year = alt.selection_single(
        fields=["YearStart"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    # ---- Top: national average trend ----
    base_trend = alt.Chart(df_map).encode(
        x=alt.X("YearStart:O", title="Year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(
            "mean(Data_Value):Q",
            title="National average obesity rate (%)",
            scale=alt.Scale(domain=[25, 35]),
        ),
    )

    line = base_trend.mark_line(color="#c0392b")

    selectors = (
        alt.Chart(df_map)
        .mark_point()
        .encode(x="YearStart:O", opacity=alt.value(0))
        .add_selection(hover_year)
    )

    points = line.mark_point(size=80, color="black").encode(
        opacity=alt.condition(hover_year, alt.value(1), alt.value(0))
    )

    rule = (
        alt.Chart(df_map)
        .mark_rule(color="gray")
        .encode(x="YearStart:O")
        .transform_filter(hover_year)
    )

    text = base_trend.mark_text(align="left", dx=5, dy=-15).encode(
        text=alt.condition(
            hover_year, "mean(Data_Value):Q", alt.value(" ")
        )
    )

    top_trend = alt.layer(line, selectors, points, rule, text).properties(
        width=700,
        height=200,
        title="Step 1: Hover over the line to select a year",
    )

    #  Bottom: US map for selected year
    states = alt.topo_feature(data.us_10m.url, "states")
    state_id_map = {
        "AL": 1,
        "AK": 2,
        "AZ": 4,
        "AR": 5,
        "CA": 6,
        "CO": 8,
        "CT": 9,
        "DE": 10,
        "FL": 12,
        "GA": 13,
        "HI": 15,
        "ID": 16,
        "IL": 17,
        "IN": 18,
        "IA": 19,
        "KS": 20,
        "KY": 21,
        "LA": 22,
        "ME": 23,
        "MD": 24,
        "MA": 25,
        "MI": 26,
        "MN": 27,
        "MS": 28,
        "MO": 29,
        "MT": 30,
        "NE": 31,
        "NV": 32,
        "NH": 33,
        "NJ": 34,
        "NM": 35,
        "NY": 36,
        "NC": 37,
        "ND": 38,
        "OH": 39,
        "OK": 40,
        "OR": 41,
        "PA": 42,
        "RI": 44,
        "SC": 45,
        "SD": 46,
        "TN": 47,
        "TX": 48,
        "UT": 49,
        "VT": 50,
        "VA": 51,
        "WA": 53,
        "WV": 54,
        "WI": 55,
        "WY": 56,
        "DC": 11,
    }
    df_map["id"] = df_map["LocationAbbr"].map(state_id_map)

    bottom_map = (
        alt.Chart(df_map)
        .mark_geoshape(stroke="white", strokeWidth=0.5)
        .encode(
            color=alt.Color(
                "Data_Value:Q",
                title="Obesity rate (%)",
                scale=alt.Scale(scheme="reds", domain=[20, 45]),
            ),
            tooltip=[
                alt.Tooltip("LocationDesc:N", title="State"),
                alt.Tooltip(
                    "Data_Value:Q", title="Obesity (%)", format=".1f"
                ),
            ],
        )
        .transform_lookup(
            lookup="id",
            from_=alt.LookupData(
                states, key="id", fields=["geometry", "type"]
            ),
        )
        .transform_filter(hover_year)
        .project("albersUsa")
        .properties(
            width=700,
            height=400,
            title="Step 2: Map updates for the selected year",
        )
    )

    return alt.vconcat(top_trend, bottom_map)


# 5. Viz 3: Income trend + growth bars
def make_income_trend(df_income_raw: pd.DataFrame) -> alt.VConcatChart:
    """
    Make a vertical layout:
      - top: trends by income group over time (national average)
      - bottom: increase since 2011 for the hovered year
    df_income_raw is expected to have:
      - YearStart
      - Stratification1 (income group)
      - Data_Value
      - Sample_Size
    """

    df = df_income_raw.copy()
    df["Sample_Size"] = pd.to_numeric(df["Sample_Size"], errors="coerce")
    df = df.dropna(subset=["Sample_Size", "Data_Value"])

    df_inc_agg = (
        df.groupby(["YearStart", "Stratification1"])
        .apply(weighted_mean)
        .reset_index(name="Data_Value")
    )

    base_2011 = (
        df_inc_agg[df_inc_agg["YearStart"] == 2011][
            ["Stratification1", "Data_Value"]
        ]
        .rename(columns={"Data_Value": "Base_2011"})
    )

    df_inc_agg = pd.merge(
        df_inc_agg, base_2011, on="Stratification1", how="left"
    )
    df_inc_agg["Growth"] = (
        df_inc_agg["Data_Value"] - df_inc_agg["Base_2011"]
    )

    income_order = [
        "Less than $15,000",
        "$15,000 - $24,999",
        "$25,000 - $34,999",
        "$35,000 - $49,999",
        "$50,000 - $74,999",
        "$75,000 or greater",
    ]

    hover_inc = alt.selection_single(
        fields=["YearStart"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    # ---- Top: income trends ----
    base_line = alt.Chart(df_inc_agg).encode(
        x=alt.X("YearStart:O", title="Year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(
            "Data_Value:Q",
            title="Obesity rate (%)",
            scale=alt.Scale(domain=[20, 45]),
        ),
        color=alt.Color(
            "Stratification1:N",
            title="Income",
            sort=income_order,
            scale=alt.Scale(scheme="tableau10"),
        ),
    )

    lines = base_line.mark_line(strokeWidth=3)

    inc_selectors = (
        alt.Chart(df_inc_agg)
        .mark_point()
        .encode(x="YearStart:O", opacity=alt.value(0))
        .add_selection(hover_inc)
    )

    rule_inc = (
        alt.Chart(df_inc_agg)
        .mark_rule(color="gray")
        .encode(x="YearStart:O")
        .transform_filter(hover_inc)
    )

    top_inc_chart = alt.layer(lines, inc_selectors, rule_inc).properties(
        width=700,
        height=250,
        title="Step 1: Obesity trends by income group",
    )

    # ---- Bottom: growth since 2011 ----
    bottom_growth = (
        alt.Chart(df_inc_agg)
        .mark_bar()
        .encode(
            x=alt.X(
                "Growth:Q",
                title="Increase since 2011 (percentage points)",
                scale=alt.Scale(domain=[0, 12]),
            ),
            y=alt.Y(
                "Stratification1:N",
                title=None,
                sort=income_order,
            ),
            color=alt.Color(
                "Stratification1:N",
                legend=None,
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=[
                alt.Tooltip(
                    "Stratification1:N", title="Income group"
                ),
                alt.Tooltip(
                    "Data_Value:Q",
                    title="Current rate",
                    format=".1f",
                ),
                alt.Tooltip(
                    "Growth:Q",
                    title="Increase since 2011",
                    format=".1f",
                ),
            ],
        )
        .transform_filter(hover_inc)
        .properties(
            width=700,
            height=250,
            title="Step 2: Who gained the most weight? (change vs 2011)",
        )
    )

    return alt.vconcat(top_inc_chart, bottom_growth)



# 6. Export charts as standalone HTML
if __name__ == "__main__":
    df_map_raw, df_age_raw, df_income_raw = load_data()

    age_chart = make_age_heatmap(df_age_raw)
    trend_map_chart = make_trend_and_map(df_map_raw)
    income_chart = make_income_trend(df_income_raw)

    age_chart.save("age_heatmap.html")
    trend_map_chart.save("trend_map.html")
    income_chart.save("income_gap.html")

    print("Saved: age_heatmap.html, trend_map.html, income_gap.html")
